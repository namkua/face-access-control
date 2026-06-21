"""
Unit tests for Face Recognition core logic.
"""
import io
import json
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sample_embedding():
    """Return a normalised 512-dim face embedding."""
    vec = np.random.rand(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def another_embedding():
    vec = np.random.rand(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def dummy_image_bytes():
    """Minimal valid JPEG bytes (1×1 white pixel)."""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x00\xff\xd9"
    )


# -----------------------------------------------------------------------
# Cosine similarity helpers
# -----------------------------------------------------------------------

class TestCosineSimilarity:
    """Tests for the cosine similarity utility."""

    def _cosine_sim(self, a, b):
        """Pure-numpy cosine similarity — matches production logic."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def test_identical_embeddings(self, sample_embedding):
        sim = self._cosine_sim(sample_embedding, sample_embedding)
        assert abs(sim - 1.0) < 1e-5, "Identical embeddings must have similarity ≈ 1.0"

    def test_orthogonal_embeddings(self):
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        sim = self._cosine_sim(a, b)
        assert abs(sim) < 1e-5, "Orthogonal embeddings must have similarity ≈ 0"

    def test_range_is_valid(self, sample_embedding, another_embedding):
        sim = self._cosine_sim(sample_embedding, another_embedding)
        assert -1.0 <= sim <= 1.0

    def test_symmetry(self, sample_embedding, another_embedding):
        assert self._cosine_sim(sample_embedding, another_embedding) == \
               self._cosine_sim(another_embedding, sample_embedding)


# -----------------------------------------------------------------------
# Face recognition service (mocked)
# -----------------------------------------------------------------------

class TestFaceRecognitionService:
    """Tests for the FaceRecognitionService — external deps mocked."""

    @pytest.fixture
    def mock_service(self, sample_embedding):
        """Return a FaceRecognitionService with mocked facenet-pytorch models."""
        with patch("api.services.face_recognition._get_mtcnn"), \
             patch("api.services.face_recognition._get_facenet"):
            from api.services.face_recognition import FaceRecognitionService

            service = FaceRecognitionService.__new__(FaceRecognitionService)
            service.model_name = "Facenet512"
            service.threshold = 0.6
            return service

    def test_calculate_similarity_cosine(self, mock_service, sample_embedding):
        """Test cosine similarity calculation with identical embeddings."""
        sim = mock_service.calculate_similarity(sample_embedding, sample_embedding, metric="cosine")
        assert abs(sim - 1.0) < 1e-5

    def test_calculate_similarity_euclidean(self, mock_service, sample_embedding, another_embedding):
        """Test euclidean distance calculation."""
        dist = mock_service.calculate_similarity(sample_embedding, another_embedding, metric="euclidean")
        assert isinstance(dist, float)
        assert dist >= 0

    def test_calculate_similarity_euclidean_l2(self, mock_service, sample_embedding, another_embedding):
        """Test L2-normalized euclidean distance calculation."""
        dist = mock_service.calculate_similarity(sample_embedding, another_embedding, metric="euclidean_l2")
        assert isinstance(dist, float)
        assert dist >= 0

    def test_calculate_similarity_unknown_metric(self, mock_service, sample_embedding):
        """Test that unknown metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            mock_service.calculate_similarity(sample_embedding, sample_embedding, metric="unknown")

    @pytest.mark.asyncio
    async def test_find_best_match_returns_match(self, mock_service, sample_embedding):
        """Test find_best_match returns correct user when above threshold."""
        database = {"alice": sample_embedding, "bob": np.random.rand(512).astype(np.float32)}
        matched_user, confidence = await mock_service.find_best_match(
            query_embedding=sample_embedding,
            database_embeddings=database,
            threshold=0.6
        )
        assert matched_user == "alice"
        assert confidence is not None
        assert confidence > 0.6

    @pytest.mark.asyncio
    async def test_find_best_match_empty_database(self, mock_service, sample_embedding):
        """Test find_best_match returns None on empty database."""
        matched_user, confidence = await mock_service.find_best_match(
            query_embedding=sample_embedding,
            database_embeddings={},
        )
        assert matched_user is None
        assert confidence is None

    @pytest.mark.asyncio
    async def test_find_best_match_below_threshold(self, mock_service, sample_embedding, another_embedding):
        """Test find_best_match returns None when below threshold."""
        database = {"bob": another_embedding}
        matched_user, confidence = await mock_service.find_best_match(
            query_embedding=sample_embedding,
            database_embeddings=database,
            threshold=0.9999
        )
        assert matched_user is None


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

class TestSettings:
    def test_default_values_are_present(self):
        from api.config import settings

        assert settings.app_name
        assert settings.api_port > 0
        assert settings.model_name

    def test_model_threshold_in_range(self):
        from api.config import settings

        assert 0.0 < settings.model_threshold <= 1.0



# -----------------------------------------------------------------------
# Tests for Module-level lazy init functions
# -----------------------------------------------------------------------
def test_get_mtcnn_lazy_init():
    import api.services.face_recognition as fr
    # Reset global state for test
    fr._mtcnn = None
    with patch("api.services.face_recognition.MTCNN") as MockMTCNN:
        mock_instance = MagicMock()
        MockMTCNN.return_value = mock_instance
        
        # First call should initialize
        mtcnn1 = fr._get_mtcnn()
        MockMTCNN.assert_called_once()
        
        # Second call should return the same instance without re-initializing
        mtcnn2 = fr._get_mtcnn()
        assert MockMTCNN.call_count == 1
        assert mtcnn1 is mtcnn2

def test_get_facenet_lazy_init():
    import api.services.face_recognition as fr
    fr._facenet = None
    with patch("api.services.face_recognition.InceptionResnetV1") as MockInception:
        mock_instance = MagicMock()
        mock_instance.eval.return_value = mock_instance
        mock_instance.to.return_value = mock_instance
        MockInception.return_value = mock_instance
        
        facenet1 = fr._get_facenet()
        MockInception.assert_called_once()
        
        facenet2 = fr._get_facenet()
        assert MockInception.call_count == 1
        assert facenet1 is facenet2

# -----------------------------------------------------------------------
# Tests for _init_triton
# -----------------------------------------------------------------------
def test_init_triton_disabled_when_no_url(monkeypatch):
    import api.services.face_recognition as fr
    fr._triton_client = None
    monkeypatch.setattr(fr.settings, "triton_url", "")
    assert fr._init_triton() is None

def test_init_triton_connected_when_server_live(monkeypatch):
    import api.services.face_recognition as fr
    fr._triton_client = None
    monkeypatch.setattr(fr.settings, "triton_url", "http://localhost:8000")
    
    with patch("tritonclient.http.InferenceServerClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.is_server_live.return_value = True
        MockClient.return_value = mock_instance
        
        client = fr._init_triton()
        assert client is mock_instance
        assert fr._triton_client is mock_instance

def test_init_triton_returns_none_when_not_live(monkeypatch):
    import api.services.face_recognition as fr
    fr._triton_client = None
    monkeypatch.setattr(fr.settings, "triton_url", "http://localhost:8000")
    
    with patch("tritonclient.http.InferenceServerClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.is_server_live.return_value = False
        MockClient.return_value = mock_instance
        
        assert fr._init_triton() is None

def test_init_triton_returns_none_on_exception(monkeypatch):
    import api.services.face_recognition as fr
    fr._triton_client = None
    monkeypatch.setattr(fr.settings, "triton_url", "http://localhost:8000")
    
    with patch("tritonclient.http.InferenceServerClient") as MockClient:
        MockClient.side_effect = Exception("Connection failed")
        assert fr._init_triton() is None

# -----------------------------------------------------------------------
# Tests for FaceRecognitionService
# -----------------------------------------------------------------------
class TestFaceRecognitionServiceExtended:
    
    @pytest.fixture
    def service(self):
        from api.services.face_recognition import FaceRecognitionService
        return FaceRecognitionService(model_name="Facenet512", threshold=0.6)

    # --- detect_face ---
    @pytest.mark.asyncio
    async def test_detect_face_success(self, service, dummy_image_bytes):
        with patch("api.services.face_recognition._get_mtcnn") as mock_get_mtcnn:
            mock_mtcnn = MagicMock()
            # Returns boxes (Nx4), probabilities (N)
            mock_mtcnn.detect.return_value = (np.array([[10, 10, 50, 50]]), np.array([0.95]))
            mock_get_mtcnn.return_value = mock_mtcnn
            
            detected, bbox = await service.detect_face(dummy_image_bytes)
            assert detected is True
            assert bbox == {"x": 10, "y": 10, "w": 40, "h": 40, "confidence": 0.95}

    @pytest.mark.asyncio
    async def test_detect_face_no_boxes(self, service, dummy_image_bytes):
        with patch("api.services.face_recognition._get_mtcnn") as mock_get_mtcnn:
            mock_mtcnn = MagicMock()
            mock_mtcnn.detect.return_value = (None, None)
            mock_get_mtcnn.return_value = mock_mtcnn
            
            detected, bbox = await service.detect_face(dummy_image_bytes)
            assert detected is False
            assert bbox is None

    @pytest.mark.asyncio
    async def test_detect_face_low_confidence(self, service, dummy_image_bytes):
        with patch("api.services.face_recognition._get_mtcnn") as mock_get_mtcnn:
            mock_mtcnn = MagicMock()
            mock_mtcnn.detect.return_value = (np.array([[10, 10, 50, 50]]), np.array([0.8]))
            mock_get_mtcnn.return_value = mock_mtcnn
            
            detected, bbox = await service.detect_face(dummy_image_bytes)
            assert detected is False
            assert bbox is None

    @pytest.mark.asyncio
    async def test_detect_face_exception(self, service, dummy_image_bytes):
        with patch("api.services.face_recognition._get_mtcnn") as mock_get_mtcnn:
            mock_get_mtcnn.side_effect = Exception("MTCNN error")
            
            detected, bbox = await service.detect_face(dummy_image_bytes)
            assert detected is False
            assert bbox is None

    # --- extract_embedding routing ---
    @pytest.mark.asyncio
    async def test_extract_embedding_uses_triton_when_available(self, service, dummy_image_bytes):
        with patch("api.services.face_recognition._init_triton") as mock_init:
            mock_init.return_value = MagicMock()
            with patch.object(service, "_extract_embedding_triton", new_callable=AsyncMock) as mock_triton:
                mock_triton.return_value = np.array([1, 2, 3])
                
                result = await service.extract_embedding(dummy_image_bytes)
                mock_triton.assert_called_once()
                np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    @pytest.mark.asyncio
    async def test_extract_embedding_uses_local_when_no_triton(self, service, dummy_image_bytes):
        with patch("api.services.face_recognition._init_triton") as mock_init:
            mock_init.return_value = None
            with patch.object(service, "_extract_embedding_local", new_callable=AsyncMock) as mock_local:
                mock_local.return_value = np.array([1, 2, 3])
                
                result = await service.extract_embedding(dummy_image_bytes)
                mock_local.assert_called_once()
                np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    # --- _extract_embedding_triton ---
    @pytest.mark.asyncio
    async def test_extract_embedding_triton_success(self, service, dummy_image_bytes):
        mock_triton_client = MagicMock()
        mock_response = MagicMock()
        # Return a batch of embeddings
        mock_response.as_numpy.return_value = np.random.rand(1, 512).astype(np.float32)
        mock_triton_client.infer.return_value = mock_response
        
        with patch("tritonclient.http.InferInput"), patch("tritonclient.http.InferRequestedOutput"):
            result = await service._extract_embedding_triton(dummy_image_bytes, mock_triton_client)
            assert result is not None
            assert result.shape == (512,)

    @pytest.mark.asyncio
    async def test_extract_embedding_triton_fallback_on_error(self, service, dummy_image_bytes):
        mock_triton_client = MagicMock()
        mock_triton_client.infer.side_effect = Exception("Triton failed")
        
        with patch.object(service, "_extract_embedding_local", new_callable=AsyncMock) as mock_local:
            mock_local.return_value = np.array([1, 2, 3])
            result = await service._extract_embedding_triton(dummy_image_bytes, mock_triton_client)
            mock_local.assert_called_once()
            np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    # --- _extract_embedding_local ---
    @pytest.mark.asyncio
    async def test_extract_embedding_local_success(self, service, dummy_image_bytes):
        with patch("api.services.face_recognition._get_mtcnn") as mock_get_mtcnn, \
             patch("api.services.face_recognition._get_facenet") as mock_get_facenet:
            
            mock_mtcnn = MagicMock()
            # MTCNN returns a tensor
            import torch
            mock_mtcnn.return_value = torch.rand(3, 160, 160)
            mock_get_mtcnn.return_value = mock_mtcnn
            
            mock_facenet = MagicMock()
            mock_facenet.return_value = torch.rand(1, 512)
            mock_get_facenet.return_value = mock_facenet
            
            result = await service._extract_embedding_local(dummy_image_bytes)
            assert result is not None
            assert result.shape == (512,)

    @pytest.mark.asyncio
    async def test_extract_embedding_local_no_face(self, service, dummy_image_bytes):
        with patch("api.services.face_recognition._get_mtcnn") as mock_get_mtcnn:
            mock_mtcnn = MagicMock()
            mock_mtcnn.return_value = None
            mock_get_mtcnn.return_value = mock_mtcnn
            
            result = await service._extract_embedding_local(dummy_image_bytes)
            assert result is None

    # --- verify_faces ---
    @pytest.mark.asyncio
    async def test_verify_faces_same_person(self, service, dummy_image_bytes, sample_embedding):
        with patch.object(service, "extract_embedding", new_callable=AsyncMock) as mock_extract:
            mock_extract.side_effect = [sample_embedding, sample_embedding]
            
            is_same, conf = await service.verify_faces(dummy_image_bytes, dummy_image_bytes)
            assert is_same is True
            assert abs(conf - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_verify_faces_different_person(self, service, dummy_image_bytes):
        with patch.object(service, "extract_embedding", new_callable=AsyncMock) as mock_extract:
            a = np.zeros(512, dtype=np.float32)
            a[0] = 1.0
            b = np.zeros(512, dtype=np.float32)
            b[1] = 1.0
            mock_extract.side_effect = [a, b]
            
            is_same, conf = await service.verify_faces(dummy_image_bytes, dummy_image_bytes)
            assert is_same is False
            assert abs(conf) < 1e-5

    @pytest.mark.asyncio
    async def test_verify_faces_no_face_detected(self, service, dummy_image_bytes, sample_embedding):
        with patch.object(service, "extract_embedding", new_callable=AsyncMock) as mock_extract:
            mock_extract.side_effect = [sample_embedding, None]
            
            is_same, conf = await service.verify_faces(dummy_image_bytes, dummy_image_bytes)
            assert is_same is False
            assert conf == 0.0

    # --- batch_extract_embeddings ---
    @pytest.mark.asyncio
    async def test_batch_extract_embeddings(self, service, dummy_image_bytes):
        with patch.object(service, "extract_embedding", new_callable=AsyncMock) as mock_extract:
            mock_extract.side_effect = [np.array([1]), None, np.array([2])]
            
            results = await service.batch_extract_embeddings([dummy_image_bytes] * 3)
            assert len(results) == 3
            np.testing.assert_array_equal(results[0], np.array([1]))
            assert results[1] is None
            np.testing.assert_array_equal(results[2], np.array([2]))

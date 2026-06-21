"""
Face recognition service using facenet-pytorch and optional NVIDIA Triton Inference Server.

When Triton is available (TRITON_URL is set and server is live), embeddings are
extracted via Triton for production-grade GPU-accelerated inference.  Otherwise,
falls back to the local facenet-pytorch pipeline.
"""
import numpy as np
from typing import Optional, Tuple, List, Dict
from PIL import Image
import io
import time
import torch
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from facenet_pytorch import MTCNN, InceptionResnetV1

from api.config import settings

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Device selection (CPU on Minikube; GPU if available)
# ---------------------------------------------------------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Lazy-loaded models (initialised once on first call)
# ---------------------------------------------------------------------------
_mtcnn: Optional[MTCNN] = None
_facenet: Optional[InceptionResnetV1] = None


def _get_mtcnn() -> MTCNN:
    """Lazily initialise MTCNN face detector."""
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            device=_device,
            post_process=True,  # returns normalised tensor
        )
        logger.info("mtcnn_initialized", device=str(_device))
    return _mtcnn


def _get_facenet() -> InceptionResnetV1:
    """Lazily initialise InceptionResnetV1 (FaceNet) embedding model."""
    global _facenet
    if _facenet is None:
        _facenet = InceptionResnetV1(pretrained="vggface2").eval().to(_device)
        logger.info("facenet_initialized", device=str(_device), pretrained="vggface2")
    return _facenet


# ---------------------------------------------------------------------------
# Optional Triton client
# ---------------------------------------------------------------------------

_triton_client = None

def _init_triton():
    """Lazily initialise a Triton HTTP client if the server is reachable."""
    global _triton_client
    if _triton_client is not None:
        return _triton_client

    if not settings.triton_url:
        logger.info("triton_disabled", reason="TRITON_URL not set")
        return None

    try:
        import tritonclient.http as httpclient
        client = httpclient.InferenceServerClient(url=settings.triton_url)
        if client.is_server_live():
            _triton_client = client
            logger.info(
                "triton_client_connected",
                url=settings.triton_url,
                model=settings.triton_model_name,
            )
            return _triton_client
        else:
            logger.warning("triton_server_not_live", url=settings.triton_url)
    except Exception as e:
        logger.warning("triton_client_init_failed", error=str(e))

    return None


class FaceRecognitionService:
    """Service for face detection and recognition using facenet-pytorch."""

    def __init__(self, model_name: str = "Facenet512", threshold: float = 0.6):
        """
        Initialize face recognition service.

        Args:
            model_name: Model identifier (kept for config compat)
            threshold: Similarity threshold for matching
        """
        self.model_name = model_name
        self.threshold = threshold
        self.triton_model_name = settings.triton_model_name
        self.triton_model_version = settings.triton_model_version

        logger.info(
            "face_recognition_service_initialized",
            model=model_name,
            threshold=threshold,
            backend="facenet-pytorch",
            triton_url=settings.triton_url or "disabled",
        )

    # ------------------------------------------------------------------
    # Face detection (MTCNN)
    # ------------------------------------------------------------------

    async def detect_face(self, image_bytes: bytes) -> Tuple[bool, Optional[Dict]]:
        """
        Detect face in image using MTCNN.

        Args:
            image_bytes: Image as bytes

        Returns:
            Tuple of (face_detected: bool, bounding_box: Optional[Dict])
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            mtcnn = _get_mtcnn()

            # Detect faces – returns boxes and probabilities
            boxes, probs = mtcnn.detect(image)

            if boxes is None or len(boxes) == 0:
                logger.warning("no_face_detected")
                return False, None

            # Take the first (highest-confidence) face
            box = boxes[0]
            prob = float(probs[0])

            if prob < 0.9:
                logger.warning("low_face_confidence", confidence=prob)
                return False, None

            bounding_box = {
                "x": int(box[0]),
                "y": int(box[1]),
                "w": int(box[2] - box[0]),
                "h": int(box[3] - box[1]),
                "confidence": prob,
            }

            return True, bounding_box

        except Exception as e:
            logger.error("face_detection_error", error=str(e))
            return False, None

    # ------------------------------------------------------------------
    # Embedding extraction – Triton → local fallback
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract face embedding from image.

        If a Triton server is available the embedding is extracted via GPU
        inference through Triton.  Otherwise, falls back to the local
        facenet-pytorch pipeline.

        Args:
            image_bytes: Image as bytes

        Returns:
            Face embedding as numpy array or None if no face detected
        """
        triton = _init_triton()

        if triton is not None:
            return await self._extract_embedding_triton(image_bytes, triton)

        return await self._extract_embedding_local(image_bytes)

    # --- Triton path -------------------------------------------------------

    async def _extract_embedding_triton(
        self, image_bytes: bytes, triton_client
    ) -> Optional[np.ndarray]:
        """Extract embedding via NVIDIA Triton Inference Server."""
        import tritonclient.http as httpclient

        try:
            start_time = time.time()

            # Preprocess image using MTCNN (detect, crop, align, normalize)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            mtcnn = _get_mtcnn()
            face_tensor = mtcnn(image)
            
            if face_tensor is None:
                logger.warning("no_face_in_image")
                return None
                
            # Add batch dimension and convert to numpy → (1, 3, 160, 160)
            img_array = face_tensor.unsqueeze(0).cpu().numpy()

            # Build Triton request
            inputs = [
                httpclient.InferInput("input", img_array.shape, "FP32"),
            ]
            inputs[0].set_data_from_numpy(img_array)

            outputs = [httpclient.InferRequestedOutput("output")]

            # Run inference
            response = triton_client.infer(
                model_name=self.triton_model_name,
                model_version=self.triton_model_version,
                inputs=inputs,
                outputs=outputs,
            )

            embedding = response.as_numpy("output").squeeze(0)

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                "embedding_extracted_triton",
                embedding_size=len(embedding),
                processing_time_ms=processing_time,
            )

            return embedding

        except Exception as e:
            logger.warning(
                "triton_inference_failed_falling_back",
                error=str(e),
            )
            # Fall back to local
            return await self._extract_embedding_local(image_bytes)

    # --- Local facenet-pytorch path ----------------------------------------

    async def _extract_embedding_local(
        self, image_bytes: bytes
    ) -> Optional[np.ndarray]:
        """Extract embedding via local facenet-pytorch pipeline."""
        try:
            start_time = time.time()

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            mtcnn = _get_mtcnn()
            facenet = _get_facenet()

            # MTCNN detects + crops + aligns face → returns (3, 160, 160) tensor
            face_tensor = mtcnn(image)

            if face_tensor is None:
                logger.warning("no_face_in_image")
                return None

            # Add batch dim → (1, 3, 160, 160) and move to device
            face_batch = face_tensor.unsqueeze(0).to(_device)

            # Extract embedding
            with torch.no_grad():
                embedding = facenet(face_batch)

            embedding_np = embedding.squeeze(0).cpu().numpy()

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                "embedding_extracted_local",
                embedding_size=len(embedding_np),
                processing_time_ms=processing_time,
            )

            return embedding_np

        except Exception as e:
            logger.error("embedding_extraction_error", error=str(e))
            raise

    # ------------------------------------------------------------------
    # Similarity calculation
    # ------------------------------------------------------------------

    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine",
    ) -> float:
        """
        Calculate similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric (cosine, euclidean, euclidean_l2)

        Returns:
            Similarity score (0-1 for cosine, lower is better for euclidean)
        """
        if metric == "cosine":
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)

        elif metric == "euclidean":
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)

        elif metric == "euclidean_l2":
            distance = np.linalg.norm(
                embedding1 / np.linalg.norm(embedding1)
                - embedding2 / np.linalg.norm(embedding2)
            )
            return float(distance)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    # ------------------------------------------------------------------
    # Best-match search
    # ------------------------------------------------------------------

    async def find_best_match(
        self,
        query_embedding: np.ndarray,
        database_embeddings: Dict[str, np.ndarray],
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Find best matching face from database.

        Args:
            query_embedding: Query face embedding
            database_embeddings: Dict of {user_id: embedding}
            threshold: Similarity threshold (uses instance threshold if None)

        Returns:
            Tuple of (matched_user_id, confidence)
        """
        if not database_embeddings:
            logger.warning("empty_database")
            return None, None

        threshold = threshold or self.threshold

        best_match = None
        best_score = -1

        for user_id, db_embedding in database_embeddings.items():
            similarity = self.calculate_similarity(
                query_embedding,
                db_embedding,
                metric="cosine",
            )

            if similarity > best_score:
                best_score = similarity
                best_match = user_id

        # Check threshold
        if best_score < threshold:
            logger.info(
                "no_match_above_threshold",
                best_score=best_score,
                threshold=threshold,
            )
            return None, best_score

        logger.info(
            "match_found",
            user_id=best_match,
            confidence=best_score,
        )

        return best_match, best_score

    # ------------------------------------------------------------------
    # Face verification
    # ------------------------------------------------------------------

    async def verify_faces(
        self,
        image1_bytes: bytes,
        image2_bytes: bytes,
    ) -> Tuple[bool, float]:
        """
        Verify if two images contain the same person.

        Args:
            image1_bytes: First image
            image2_bytes: Second image

        Returns:
            Tuple of (is_same_person: bool, confidence: float)
        """
        try:
            embedding1 = await self.extract_embedding(image1_bytes)
            embedding2 = await self.extract_embedding(image2_bytes)

            if embedding1 is None or embedding2 is None:
                return False, 0.0

            similarity = self.calculate_similarity(embedding1, embedding2)
            is_same = similarity >= self.threshold

            logger.info(
                "face_verification",
                is_same=is_same,
                similarity=similarity,
            )

            return is_same, similarity

        except Exception as e:
            logger.error("face_verification_error", error=str(e))
            return False, 0.0

    # ------------------------------------------------------------------
    # Batch extraction via Triton
    # ------------------------------------------------------------------

    async def batch_extract_embeddings(
        self, images_bytes: List[bytes]
    ) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings for multiple images.

        Uses Triton batch inference when available, otherwise falls
        back to sequential local calls.

        Args:
            images_bytes: List of image byte arrays

        Returns:
            List of embeddings (None for failed extractions)
        """
        results = []
        for img_bytes in images_bytes:
            emb = await self.extract_embedding(img_bytes)
            results.append(emb)
        return results


# Global instance
face_recognition_service = FaceRecognitionService(
    model_name=settings.model_name,
    threshold=settings.model_threshold,
)

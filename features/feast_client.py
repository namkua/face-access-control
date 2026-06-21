"""
Feast feature store client for retrieving features.

Updated to use feature views from feast_features.py:
  - employee_face_embeddings (entity: employee, join_key: employee_id)
  - employee_profile (entity: employee)
  - employee_checkin_stats (entity: employee)
"""
import os
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from feast import FeatureStore
import structlog

logger = structlog.get_logger()


class FeastFeatureStoreClient:
    """Client for interacting with Feast feature store."""

    def __init__(self, repo_path: str = None):
        """
        Initialize Feast client.

        Args:
            repo_path: Path to Feast repository (defaults to FEAST_REPO_PATH env var or /feast_repo)
        """
        if repo_path is None:
            repo_path = os.environ.get("FEAST_REPO_PATH", "/feast_repo")
        self.store = FeatureStore(repo_path=repo_path)
        logger.info("feast_client_initialized", repo_path=repo_path)

    async def get_online_features(
        self,
        employee_ids: List[str],
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Retrieve online features for employees.

        Args:
            employee_ids: List of employee IDs
            features: List of feature names to retrieve

        Returns:
            DataFrame with features
        """
        if features is None:
            features = [
                "employee_face_embeddings:embedding",
                "employee_face_embeddings:embedding_model",
                "employee_face_embeddings:registration_confidence",
            ]

        # Retrieve features
        feature_vector = self.store.get_online_features(
            features=features,
            entity_rows=[{"employee_id": eid} for eid in employee_ids],
        )

        result_df = feature_vector.to_df()

        logger.info(
            "features_retrieved",
            num_employees=len(employee_ids),
            num_features=len(features),
        )

        return result_df

    async def get_embeddings_for_matching(
        self,
        employee_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get face embeddings for all active employees.

        Args:
            employee_ids: Optional list of specific employee IDs

        Returns:
            Dict of {employee_id: embedding_vector}
        """
        if employee_ids is None:
            logger.warning("get_all_employees_not_implemented")
            return {}

        features_df = await self.get_online_features(
            employee_ids=employee_ids,
            features=["employee_face_embeddings:embedding"],
        )

        embeddings = {}
        for _, row in features_df.iterrows():
            employee_id = row["employee_id"]
            # Depending on Feast configuration/version, it might return full_name or short_name
            embedding = row.get("employee_face_embeddings__embedding") or row.get("embedding")
            if embedding is not None and isinstance(embedding, (list, np.ndarray)):
                embeddings[employee_id] = np.array(embedding, dtype=np.float32)

        return embeddings

    def push_embedding(
        self,
        employee_id: str,
        embedding: np.ndarray,
        model: str = "facenet",
        version: int = 1,
        confidence: float = 1.0,
    ) -> None:
        """
        Push a new embedding to the Feast online and offline stores.

        Args:
            employee_id: The employee's unique ID
            embedding: The face embedding vector
            model: The model used to generate the embedding
            version: The model version
            confidence: The confidence score of the face registration
        """
        from datetime import datetime

        df = pd.DataFrame(
            {
                "employee_id": [employee_id],
                "embedding": [embedding.tolist()],
                "embedding_model": [model],
                "embedding_version": [version],
                "image_quality_score": [1.0],
                "num_faces_detected": [1],
                "registration_confidence": [confidence],
                "event_timestamp": [datetime.utcnow()],
                "created_at": [datetime.utcnow()],
            }
        )

        self.store.push("employee_embedding_push", df)
        
        logger.info(
            "embedding_pushed_to_feast",
            employee_id=employee_id,
            model=model,
            version=version,
        )

    async def materialize_features(
        self,
        start_date: str,
        end_date: str,
    ) -> None:
        """
        Materialize features from offline to online store.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        from datetime import datetime

        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        self.store.materialize(start_date=start, end_date=end)

        logger.info(
            "features_materialized",
            start_date=start_date,
            end_date=end_date,
        )


# Lazy global instance — avoid instantiation at import time (breaks `feast apply`)
_feast_client: FeastFeatureStoreClient = None


def get_feast_client() -> FeastFeatureStoreClient:
    """Return (and lazily create) the global Feast client."""
    global _feast_client
    if _feast_client is None:
        _feast_client = FeastFeatureStoreClient()
    return _feast_client

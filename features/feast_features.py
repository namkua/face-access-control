"""
Feast Feature Store Definitions for Face Recognition System.

Defines feature views for face embeddings using:
- Online store: Redis (low-latency inference)
- Offline store: PostgreSQL (historical data, model retraining)
"""
from datetime import timedelta
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    PushSource,
)
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.types import Array, Float32, Int64, String


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

employee = Entity(
    name="employee",
    join_keys=["employee_id"],
    description="An employee identified by their unique ID",
)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

# Offline batch source: PostgreSQL table in the 'feast' schema.
# This table is created by feast_init.sh before `feast apply` runs.
# Used by Feast for historical feature retrieval and batch materialization
# during model retraining.
employee_embeddings_source = PostgreSQLSource(
    name="employee_embeddings_pg_source",
    query="""
        SELECT 
            id::text AS employee_id, 
            (SELECT array_agg(e::text::real) FROM json_array_elements(face_embedding) e) AS embedding,
            'facenet' AS embedding_model,
            1 AS embedding_version,
            1.0::real AS image_quality_score,
            1::bigint AS num_faces_detected,
            1.0::real AS registration_confidence,
            updated_at AS event_timestamp,
            created_at AS created_at
        FROM public.users 
        WHERE face_embedding IS NOT NULL
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
)

# Push source for real-time feature updates (e.g., after onboarding).
# Calling store.push() writes simultaneously to:
#   - Redis online store  (for low-latency inference)
#   - PostgreSQL offline store (for historical retrieval / retraining)
embedding_push_source = PushSource(
    name="employee_embedding_push",
    batch_source=employee_embeddings_source,
)

# ---------------------------------------------------------------------------
# Feature Views
# ---------------------------------------------------------------------------

employee_embedding_fv = FeatureView(
    name="employee_face_embeddings",
    entities=[employee],
    ttl=timedelta(days=365),
    schema=[
        Field(name="embedding", dtype=Array(Float32)),      # 512-dim FaceNet vector
        Field(name="embedding_model", dtype=String),        # e.g. "facenet", "arcface"
        Field(name="embedding_version", dtype=Int64),       # model version tag
        Field(name="image_quality_score", dtype=Float32),
        Field(name="num_faces_detected", dtype=Int64),
        Field(name="registration_confidence", dtype=Float32),
    ],
    online=True,
    source=embedding_push_source,
    tags={"team": "mlops", "domain": "face-recognition"},
)

# ---------------------------------------------------------------------------
# Feature Services
# ---------------------------------------------------------------------------

face_recognition_service = FeatureService(
    name="face_recognition_feature_service",
    features=[
        employee_embedding_fv,
    ],
    description="Features needed for real-time face recognition inference",
)

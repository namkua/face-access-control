"""Application configuration using Pydantic Settings."""
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = Field(default="face-recognition-mlops", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    cors_origins: List[str] = Field(default=["http://localhost:3000"], env="CORS_ORIGINS")
    
    # Database
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="face_recognition", env="POSTGRES_DB")
    postgres_user: str = Field(default="admin", env="POSTGRES_USER")
    postgres_password: str = Field(default="changeme123", env="POSTGRES_PASSWORD")
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    
    # Kafka
    kafka_bootstrap_servers: str = Field(default="localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_topic_access_logs: str = Field(default="access-logs", env="KAFKA_TOPIC_ACCESS_LOGS")
    kafka_topic_events: str = Field(default="events", env="KAFKA_TOPIC_EVENTS")
    kafka_topic_alerts: str = Field(default="alerts", env="KAFKA_TOPIC_ALERTS")
    
    # MinIO
    minio_endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    minio_bucket_raw: str = Field(default="raw-images", env="MINIO_BUCKET_RAW")
    minio_bucket_processed: str = Field(default="processed-data", env="MINIO_BUCKET_PROCESSED")
    minio_secure: bool = Field(default=False, env="MINIO_SECURE")
    
    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    
    # Monitoring
    api_prometheus_port: int = Field(default=9090, env="API_PROMETHEUS_PORT")
    jaeger_agent_host: str = Field(default="localhost", env="JAEGER_AGENT_HOST")
    jaeger_agent_port: int = Field(default=6831, env="JAEGER_AGENT_PORT")
    
    # Triton
    triton_url: str = Field(default="", env="TRITON_URL")
    triton_model_name: str = Field(default="facenet", env="TRITON_MODEL_NAME")
    triton_model_version: str = Field(default="1", env="TRITON_MODEL_VERSION")
    
    # Feast
    feast_server_url: str = Field(default="", env="FEAST_SERVER_URL")
    feast_enabled: bool = Field(default=True, env="FEAST_ENABLED")
    
    # Model
    model_name: str = Field(default="facenet", env="MODEL_NAME")
    model_threshold: float = Field(default=0.6, env="MODEL_THRESHOLD")
    max_image_size_mb: int = Field(default=10, env="MAX_IMAGE_SIZE_MB")
    allowed_extensions: List[str] = Field(default=["jpg", "jpeg", "png"], env="ALLOWED_EXTENSIONS")
    
    # GCP
    gcp_project_id: Optional[str] = Field(default=None, env="GCP_PROJECT_ID")
    gcp_region: str = Field(default="us-central1", env="GCP_REGION")
    gcp_zone: str = Field(default="us-central1-a", env="GCP_ZONE")
    gke_cluster_name: str = Field(default="face-recognition-cluster", env="GKE_CLUSTER_NAME")
    gcs_bucket: Optional[str] = Field(default=None, env="GCS_BUCKET")
    
    @validator("database_url", pre=True, always=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        """Construct database URL if not provided."""
        if v:
            return v
        return (
            f"postgresql+asyncpg://{values.get('postgres_user')}:"
            f"{values.get('postgres_password')}@{values.get('postgres_host')}:"
            f"{values.get('postgres_port')}/{values.get('postgres_db')}"
        )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        protected_namespaces = ()


# Global settings instance
settings = Settings()

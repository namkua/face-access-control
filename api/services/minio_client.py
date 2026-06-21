"""MinIO client service for object storage."""
from minio import Minio
from minio.error import S3Error
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import io
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from api.config import settings

logger = structlog.get_logger()


class MinIOService:
    """Service for interacting with MinIO object storage."""
    
    def __init__(self):
        """Initialize MinIO client."""
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        self.bucket_raw = settings.minio_bucket_raw
        self.bucket_processed = settings.minio_bucket_processed
        
        # Ensure buckets exist
        self._ensure_buckets()
        
        logger.info(
            "minio_service_initialized",
            endpoint=settings.minio_endpoint,
            buckets=[self.bucket_raw, self.bucket_processed]
        )
    
    def _ensure_buckets(self):
        """Create buckets if they don't exist."""
        for bucket_name in [self.bucket_raw, self.bucket_processed]:
            try:
                if not self.client.bucket_exists(bucket_name):
                    self.client.make_bucket(bucket_name)
                    logger.info("bucket_created", bucket=bucket_name)
            except S3Error as e:
                logger.error("bucket_creation_error", bucket=bucket_name, error=str(e))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def upload_file(
        self,
        file_bytes: bytes,
        object_name: str,
        bucket: Optional[str] = None,
        content_type: str = "image/jpeg",
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload file to MinIO.
        
        Args:
            file_bytes: File content as bytes
            object_name: Object name in bucket
            bucket: Bucket name (defaults to raw bucket)
            content_type: MIME type
            metadata: Optional metadata tags
            
        Returns:
            True if successful
        """
        bucket = bucket or self.bucket_raw
        
        try:
            # Convert bytes to file-like object
            file_data = io.BytesIO(file_bytes)
            file_size = len(file_bytes)
            
            # Upload
            self.client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=file_data,
                length=file_size,
                content_type=content_type,
                metadata=metadata or {}
            )
            
            logger.info(
                "file_uploaded",
                bucket=bucket,
                object_name=object_name,
                size_bytes=file_size
            )
            
            return True
            
        except S3Error as e:
            logger.error(
                "minio_upload_error",
                bucket=bucket,
                object_name=object_name,
                error=str(e)
            )
            return False
    
    async def download_file(
        self,
        object_name: str,
        bucket: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Download file from MinIO.
        
        Args:
            object_name: Object name in bucket
            bucket: Bucket name (defaults to raw bucket)
            
        Returns:
            File content as bytes or None if error
        """
        bucket = bucket or self.bucket_raw
        
        try:
            response = self.client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            
            logger.info(
                "file_downloaded",
                bucket=bucket,
                object_name=object_name,
                size_bytes=len(data)
            )
            
            return data
            
        except S3Error as e:
            logger.error(
                "minio_download_error",
                bucket=bucket,
                object_name=object_name,
                error=str(e)
            )
            return None
    
    async def delete_file(
        self,
        object_name: str,
        bucket: Optional[str] = None
    ) -> bool:
        """
        Delete file from MinIO.
        
        Args:
            object_name: Object name in bucket
            bucket: Bucket name (defaults to raw bucket)
            
        Returns:
            True if successful
        """
        bucket = bucket or self.bucket_raw
        
        try:
            self.client.remove_object(bucket, object_name)
            logger.info("file_deleted", bucket=bucket, object_name=object_name)
            return True
            
        except S3Error as e:
            logger.error(
                "minio_delete_error",
                bucket=bucket,
                object_name=object_name,
                error=str(e)
            )
            return False
    
    async def get_presigned_url(
        self,
        object_name: str,
        bucket: Optional[str] = None,
        expires: timedelta = timedelta(hours=1)
    ) -> Optional[str]:
        """
        Generate presigned URL for temporary access.
        
        Args:
            object_name: Object name in bucket
            bucket: Bucket name (defaults to raw bucket)
            expires: URL expiration time
            
        Returns:
            Presigned URL or None if error
        """
        bucket = bucket or self.bucket_raw
        
        try:
            url = self.client.presigned_get_object(
                bucket_name=bucket,
                object_name=object_name,
                expires=expires
            )
            
            logger.info(
                "presigned_url_generated",
                bucket=bucket,
                object_name=object_name,
                expires_seconds=expires.total_seconds()
            )
            
            return url
            
        except S3Error as e:
            logger.error(
                "presigned_url_error",
                bucket=bucket,
                object_name=object_name,
                error=str(e)
            )
            return None
    
    async def list_objects(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None
    ) -> list:
        """
        List objects in bucket.
        
        Args:
            bucket: Bucket name (defaults to raw bucket)
            prefix: Filter by prefix
            
        Returns:
            List of object names
        """
        bucket = bucket or self.bucket_raw
        
        try:
            objects = self.client.list_objects(bucket, prefix=prefix)
            object_names = [obj.object_name for obj in objects]
            
            logger.info(
                "objects_listed",
                bucket=bucket,
                prefix=prefix,
                count=len(object_names)
            )
            
            return object_names
            
        except S3Error as e:
            logger.error(
                "list_objects_error",
                bucket=bucket,
                error=str(e)
            )
            return []


# Global instance
minio_service = MinIOService()

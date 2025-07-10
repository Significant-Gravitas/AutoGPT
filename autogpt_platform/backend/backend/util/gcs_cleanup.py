"""
GCS cleanup utility for removing expired files.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from google.cloud import storage
from google.cloud.storage.blob import Blob

from backend.util.settings import Config

logger = logging.getLogger(__name__)


class GCSCleanupManager:
    """
    Manager for cleaning up expired files in Google Cloud Storage.
    """
    
    def __init__(self):
        self.config = Config()
        self.storage_client = None
        self.bucket = None
        self._init_gcs_client()
    
    def _init_gcs_client(self):
        """Initialize Google Cloud Storage client and bucket."""
        try:
            if not self.config.media_gcs_bucket_name:
                raise ValueError("GCS bucket name not configured.")
            
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.config.media_gcs_bucket_name)
            
            logger.info(f"GCS cleanup manager initialized for bucket: {self.config.media_gcs_bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCS cleanup manager: {e}")
            raise
    
    def _is_blob_expired(self, blob: Blob) -> bool:
        """Check if a blob has expired based on its metadata."""
        try:
            if not blob.metadata:
                return False
            
            expires_at_str = blob.metadata.get("expires_at")
            if not expires_at_str:
                return False
            
            # Parse expiration time
            expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            
            return now > expires_at
        
        except Exception as e:
            logger.warning(f"Failed to check expiration for blob {blob.name}: {e}")
            return False
    
    def cleanup_expired_files(self, prefix: str = "autogpt-temp/") -> Dict[str, Any]:
        """
        Clean up expired files with the given prefix.
        
        Args:
            prefix: GCS path prefix to search for expired files
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            logger.info(f"Starting cleanup of expired files with prefix: {prefix}")
            
            deleted_files = []
            errors = []
            total_checked = 0
            
            # List all blobs with the prefix
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                total_checked += 1
                
                try:
                    # Check if blob is expired
                    if self._is_blob_expired(blob):
                        # Delete the expired blob
                        blob.delete()
                        deleted_files.append({
                            "path": blob.name,
                            "size": blob.size,
                            "deleted_at": datetime.utcnow().isoformat()
                        })
                        logger.info(f"Deleted expired file: {blob.name}")
                
                except Exception as e:
                    error_msg = f"Failed to process blob {blob.name}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            result = {
                "total_checked": total_checked,
                "deleted_count": len(deleted_files),
                "deleted_files": deleted_files,
                "errors": errors,
                "cleanup_time": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Cleanup completed. Checked: {total_checked}, Deleted: {len(deleted_files)}, Errors: {len(errors)}")
            
            return result
        
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise
    
    def get_expired_files_info(self, prefix: str = "autogpt-temp/") -> List[Dict[str, Any]]:
        """
        Get information about expired files without deleting them.
        
        Args:
            prefix: GCS path prefix to search for expired files
            
        Returns:
            List of dictionaries with expired file information
        """
        try:
            expired_files = []
            
            # List all blobs with the prefix
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                if self._is_blob_expired(blob):
                    expired_files.append({
                        "path": blob.name,
                        "size": blob.size,
                        "content_type": blob.content_type,
                        "time_created": blob.time_created.isoformat() if blob.time_created else None,
                        "expires_at": blob.metadata.get("expires_at") if blob.metadata else None
                    })
            
            return expired_files
        
        except Exception as e:
            logger.error(f"Failed to get expired files info: {e}")
            raise
    
    def cleanup_old_files_by_age(self, prefix: str = "autogpt-temp/", max_age_hours: int = 168) -> Dict[str, Any]:
        """
        Clean up files older than the specified age, regardless of metadata.
        
        Args:
            prefix: GCS path prefix to search for old files
            max_age_hours: Maximum age in hours (default 168 = 7 days)
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            logger.info(f"Starting cleanup of files older than {max_age_hours} hours with prefix: {prefix}")
            
            deleted_files = []
            errors = []
            total_checked = 0
            
            cutoff_time = datetime.now(timezone.utc) - timezone.timedelta(hours=max_age_hours)
            
            # List all blobs with the prefix
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                total_checked += 1
                
                try:
                    # Check if blob is older than cutoff time
                    if blob.time_created and blob.time_created < cutoff_time:
                        # Delete the old blob
                        blob.delete()
                        deleted_files.append({
                            "path": blob.name,
                            "size": blob.size,
                            "created_at": blob.time_created.isoformat(),
                            "deleted_at": datetime.utcnow().isoformat()
                        })
                        logger.info(f"Deleted old file: {blob.name}")
                
                except Exception as e:
                    error_msg = f"Failed to process blob {blob.name}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            result = {
                "total_checked": total_checked,
                "deleted_count": len(deleted_files),
                "deleted_files": deleted_files,
                "errors": errors,
                "cleanup_time": datetime.utcnow().isoformat(),
                "max_age_hours": max_age_hours
            }
            
            logger.info(f"Age-based cleanup completed. Checked: {total_checked}, Deleted: {len(deleted_files)}, Errors: {len(errors)}")
            
            return result
        
        except Exception as e:
            logger.error(f"Age-based cleanup failed: {e}")
            raise


def cleanup_expired_gcs_files(prefix: str = "autogpt-temp/") -> Dict[str, Any]:
    """
    Convenience function to clean up expired GCS files.
    
    Args:
        prefix: GCS path prefix to search for expired files
        
    Returns:
        Dictionary with cleanup statistics
    """
    cleanup_manager = GCSCleanupManager()
    return cleanup_manager.cleanup_expired_files(prefix)


def cleanup_old_gcs_files(prefix: str = "autogpt-temp/", max_age_hours: int = 168) -> Dict[str, Any]:
    """
    Convenience function to clean up old GCS files by age.
    
    Args:
        prefix: GCS path prefix to search for old files
        max_age_hours: Maximum age in hours (default 168 = 7 days)
        
    Returns:
        Dictionary with cleanup statistics
    """
    cleanup_manager = GCSCleanupManager()
    return cleanup_manager.cleanup_old_files_by_age(prefix, max_age_hours)
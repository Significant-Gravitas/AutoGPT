"""GCS cleanup monitoring module."""

import logging
from datetime import datetime

from backend.notifications.notifications import NotificationManagerClient
from backend.util.gcs_cleanup import cleanup_expired_gcs_files, cleanup_old_gcs_files
from backend.util.service import get_service_client

logger = logging.getLogger(__name__)


class GCSCleanupMonitor:
    """Monitor and clean up expired GCS files."""
    
    def __init__(self):
        self.notification_client = get_service_client(NotificationManagerClient)
    
    def cleanup_expired_files(self) -> str:
        """Clean up expired GCS files and report results."""
        try:
            logger.info("Starting GCS expired files cleanup")
            
            # Clean up expired files
            result = cleanup_expired_gcs_files()
            
            # Log results
            total_checked = result.get("total_checked", 0)
            deleted_count = result.get("deleted_count", 0)
            errors = result.get("errors", [])
            
            if deleted_count > 0:
                message = (
                    f"🧹 GCS Cleanup Report:\n"
                    f"• Files checked: {total_checked}\n"
                    f"• Expired files deleted: {deleted_count}\n"
                    f"• Errors: {len(errors)}"
                )
                
                if errors:
                    message += f"\n\n⚠️ Cleanup errors:\n" + "\n".join(f"• {error}" for error in errors[:5])
                    if len(errors) > 5:
                        message += f"\n• ... and {len(errors) - 5} more errors"
                
                # Send notification for significant cleanup activity
                if deleted_count > 10 or errors:
                    self.notification_client.discord_system_alert(message)
                
                logger.info(f"GCS cleanup completed: {deleted_count} files deleted, {len(errors)} errors")
                return f"Cleaned up {deleted_count} expired files"
            else:
                logger.info(f"GCS cleanup completed: no expired files found ({total_checked} files checked)")
                return f"No expired files found ({total_checked} files checked)"
        
        except Exception as e:
            error_msg = f"GCS cleanup failed: {e}"
            logger.error(error_msg)
            
            # Send error notification
            self.notification_client.discord_system_alert(f"❌ GCS Cleanup Error: {error_msg}")
            return error_msg
    
    def cleanup_old_files(self, max_age_hours: int = 168) -> str:
        """Clean up old GCS files by age and report results."""
        try:
            logger.info(f"Starting GCS old files cleanup (max age: {max_age_hours} hours)")
            
            # Clean up old files
            result = cleanup_old_gcs_files(max_age_hours=max_age_hours)
            
            # Log results
            total_checked = result.get("total_checked", 0)
            deleted_count = result.get("deleted_count", 0)
            errors = result.get("errors", [])
            
            if deleted_count > 0:
                message = (
                    f"🧹 GCS Age-based Cleanup Report:\n"
                    f"• Files checked: {total_checked}\n"
                    f"• Old files deleted: {deleted_count}\n"
                    f"• Max age: {max_age_hours} hours\n"
                    f"• Errors: {len(errors)}"
                )
                
                if errors:
                    message += f"\n\n⚠️ Cleanup errors:\n" + "\n".join(f"• {error}" for error in errors[:5])
                    if len(errors) > 5:
                        message += f"\n• ... and {len(errors) - 5} more errors"
                
                # Send notification for significant cleanup activity
                if deleted_count > 10 or errors:
                    self.notification_client.discord_system_alert(message)
                
                logger.info(f"GCS age-based cleanup completed: {deleted_count} files deleted, {len(errors)} errors")
                return f"Cleaned up {deleted_count} old files"
            else:
                logger.info(f"GCS age-based cleanup completed: no old files found ({total_checked} files checked)")
                return f"No old files found ({total_checked} files checked)"
        
        except Exception as e:
            error_msg = f"GCS age-based cleanup failed: {e}"
            logger.error(error_msg)
            
            # Send error notification
            self.notification_client.discord_system_alert(f"❌ GCS Age-based Cleanup Error: {error_msg}")
            return error_msg


def cleanup_expired_gcs_files_job(**kwargs):
    """Scheduled job to clean up expired GCS files."""
    monitor = GCSCleanupMonitor()
    return monitor.cleanup_expired_files()


def cleanup_old_gcs_files_job(**kwargs):
    """Scheduled job to clean up old GCS files."""
    max_age_hours = kwargs.get("max_age_hours", 168)  # Default 7 days
    monitor = GCSCleanupMonitor()
    return monitor.cleanup_old_files(max_age_hours)
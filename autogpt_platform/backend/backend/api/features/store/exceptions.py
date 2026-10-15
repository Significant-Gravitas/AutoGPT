from backend.util.exceptions import NotFoundError


class MediaUploadError(ValueError):
    """Base exception for media upload errors"""


class InvalidFileTypeError(MediaUploadError):
    """Raised when file type is not supported"""


class FileSizeTooLargeError(MediaUploadError):
    """Raised when file size exceeds maximum limit"""


class FileReadError(MediaUploadError):
    """Raised when there's an error reading the file"""


class StorageConfigError(MediaUploadError):
    """Raised when storage configuration is invalid"""


class StorageUploadError(MediaUploadError):
    """Raised when upload to storage fails"""


class VirusDetectedError(MediaUploadError):
    """Raised when a virus is detected in uploaded file"""

    def __init__(self, threat_name: str, message: str | None = None):
        self.threat_name = threat_name
        super().__init__(message or f"Virus detected: {threat_name}")


class VirusScanError(MediaUploadError):
    """Raised when virus scanning fails"""


class StoreError(ValueError):
    """Base exception for store-related errors"""


class CreatorNotFoundError(NotFoundError):
    """Raised when a creator is not found"""


class ListingExistsError(StoreError):
    """Raised when trying to create a listing that already exists"""


class ProfileNotFoundError(NotFoundError):
    """Raised when a profile is not found"""


class ListingNotFoundError(NotFoundError):
    """Raised when a store listing is not found"""


class SubmissionNotFoundError(NotFoundError):
    """Raised when a submission is not found"""


class InvalidOperationError(StoreError):
    """Raised when an operation is not valid for the current state"""


class UnauthorizedError(StoreError):
    """Raised when a user is not authorized to perform an action"""


class SlugAlreadyInUseError(StoreError):
    """Raised when a slug is already in use by another agent owned by the user"""

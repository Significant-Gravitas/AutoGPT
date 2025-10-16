from backend.util.exceptions import NotFoundError


class MediaUploadError(ValueError):
    """Base exception for media upload errors"""

    pass


class InvalidFileTypeError(MediaUploadError):
    """Raised when file type is not supported"""

    pass


class FileSizeTooLargeError(MediaUploadError):
    """Raised when file size exceeds maximum limit"""

    pass


class FileReadError(MediaUploadError):
    """Raised when there's an error reading the file"""

    pass


class StorageConfigError(MediaUploadError):
    """Raised when storage configuration is invalid"""

    pass


class StorageUploadError(MediaUploadError):
    """Raised when upload to storage fails"""

    pass


class VirusDetectedError(MediaUploadError):
    """Raised when a virus is detected in uploaded file"""

    def __init__(self, threat_name: str, message: str | None = None):
        self.threat_name = threat_name
        super().__init__(message or f"Virus detected: {threat_name}")


class VirusScanError(MediaUploadError):
    """Raised when virus scanning fails"""

    pass


class StoreError(ValueError):
    """Base exception for store-related errors"""

    pass


class AgentNotFoundError(NotFoundError):
    """Raised when an agent is not found"""

    pass


class CreatorNotFoundError(NotFoundError):
    """Raised when a creator is not found"""

    pass


class ListingExistsError(StoreError):
    """Raised when trying to create a listing that already exists"""

    pass


class ProfileNotFoundError(NotFoundError):
    """Raised when a profile is not found"""

    pass


class ListingNotFoundError(NotFoundError):
    """Raised when a store listing is not found"""

    pass


class SubmissionNotFoundError(NotFoundError):
    """Raised when a submission is not found"""

    pass


class InvalidOperationError(StoreError):
    """Raised when an operation is not valid for the current state"""

    pass


class UnauthorizedError(StoreError):
    """Raised when a user is not authorized to perform an action"""

    pass


class SlugAlreadyInUseError(StoreError):
    """Raised when a slug is already in use by another agent owned by the user"""

    pass

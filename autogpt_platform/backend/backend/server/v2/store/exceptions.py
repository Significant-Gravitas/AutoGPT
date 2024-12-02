class MediaUploadError(Exception):
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


class StoreError(Exception):
    """Base exception for store-related errors"""

    pass


class AgentNotFoundError(StoreError):
    """Raised when an agent is not found"""

    pass


class CreatorNotFoundError(StoreError):
    """Raised when a creator is not found"""

    pass


class ListingExistsError(StoreError):
    """Raised when trying to create a listing that already exists"""

    pass


class DatabaseError(StoreError):
    """Raised when there is an error interacting with the database"""

    pass


class ProfileNotFoundError(StoreError):
    """Raised when a profile is not found"""

    pass


class SubmissionNotFoundError(StoreError):
    """Raised when a submission is not found"""

    pass

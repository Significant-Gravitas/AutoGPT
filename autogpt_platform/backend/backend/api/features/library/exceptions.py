class FolderValidationError(Exception):
    """Raised when folder operations fail validation."""

    pass


class FolderAlreadyExistsError(FolderValidationError):
    """Raised when a folder with the same name already exists in the location."""

    pass

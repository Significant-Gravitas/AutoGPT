class FolderValidationError(Exception):
    """Raised when folder operations fail validation."""

    pass


class FolderAlreadyExistsError(FolderValidationError):
    """Raised when a folder with the same name already exists in the location."""

    pass


class LibraryAgentInAnotherOrganizationError(Exception):
    """The user already has this agent in their library, in another organization.

    A library entry is unique per user, graph and version, so a second one
    can't be made, and the existing one isn't this organization's to restore
    or return.
    """

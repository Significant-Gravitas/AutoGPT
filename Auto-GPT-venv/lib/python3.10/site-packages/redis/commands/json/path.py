class Path:
    """This class represents a path in a JSON value."""

    strPath = ""

    @staticmethod
    def root_path():
        """Return the root path's string representation."""
        return "."

    def __init__(self, path):
        """Make a new path based on the string representation in `path`."""
        self.strPath = path

    def __repr__(self):
        return self.strPath

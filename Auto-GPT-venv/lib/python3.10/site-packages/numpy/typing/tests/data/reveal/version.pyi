import numpy.version

reveal_type(numpy.version.version)  # E: str
reveal_type(numpy.version.__version__)  # E: str
reveal_type(numpy.version.full_version)  # E: str
reveal_type(numpy.version.git_revision)  # E: str
reveal_type(numpy.version.release)  # E: bool
reveal_type(numpy.version.short_version)  # E: str

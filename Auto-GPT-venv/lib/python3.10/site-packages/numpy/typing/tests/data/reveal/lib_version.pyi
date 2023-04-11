from numpy.lib import NumpyVersion

version = NumpyVersion("1.8.0")

reveal_type(version.vstring)  # E: str
reveal_type(version.version)  # E: str
reveal_type(version.major)  # E: int
reveal_type(version.minor)  # E: int
reveal_type(version.bugfix)  # E: int
reveal_type(version.pre_release)  # E: str
reveal_type(version.is_devversion)  # E: bool

reveal_type(version == version)  # E: bool
reveal_type(version != version)  # E: bool
reveal_type(version < "1.8.0")  # E: bool
reveal_type(version <= version)  # E: bool
reveal_type(version > version)  # E: bool
reveal_type(version >= "1.8.0")  # E: bool

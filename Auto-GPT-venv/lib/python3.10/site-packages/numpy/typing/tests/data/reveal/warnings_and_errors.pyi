import numpy as np

reveal_type(np.ModuleDeprecationWarning())  # E: ModuleDeprecationWarning
reveal_type(np.VisibleDeprecationWarning())  # E: VisibleDeprecationWarning
reveal_type(np.ComplexWarning())  # E: ComplexWarning
reveal_type(np.RankWarning())  # E: RankWarning
reveal_type(np.TooHardError())  # E: TooHardError
reveal_type(np.AxisError("test"))  # E: AxisError
reveal_type(np.AxisError(5, 1))  # E: AxisError

import numpy as np

np.deprecate(1)  # E: No overload variant

np.deprecate_with_doc(1)  # E: incompatible type

np.byte_bounds(1)  # E: incompatible type

np.who(1)  # E: incompatible type

np.lookfor(None)  # E: incompatible type

np.safe_eval(None)  # E: incompatible type

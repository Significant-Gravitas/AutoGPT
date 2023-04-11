import numpy as np

i8 = np.int64()
i4 = np.int32()
u8 = np.uint64()
b_ = np.bool_()
i = int()

f8 = np.float64()

b_ >> f8  # E: No overload variant
i8 << f8  # E: No overload variant
i | f8  # E: Unsupported operand types
i8 ^ f8  # E: No overload variant
u8 & f8  # E: No overload variant
~f8  # E: Unsupported operand type

# mypys' error message for `NoReturn` is unfortunately pretty bad
# TODO: Re-enable this once we add support for numerical precision for `number`s
# a = u8 | 0  # E: Need type annotation

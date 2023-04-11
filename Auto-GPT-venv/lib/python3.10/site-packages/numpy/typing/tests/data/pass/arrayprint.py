import numpy as np

AR = np.arange(10)
AR.setflags(write=False)

with np.printoptions():
    np.set_printoptions(
        precision=1,
        threshold=2,
        edgeitems=3,
        linewidth=4,
        suppress=False,
        nanstr="Bob",
        infstr="Bill",
        formatter={},
        sign="+",
        floatmode="unique",
    )
    np.get_printoptions()
    str(AR)

    np.array2string(
        AR,
        max_line_width=5,
        precision=2,
        suppress_small=True,
        separator=";",
        prefix="test",
        threshold=5,
        floatmode="fixed",
        suffix="?",
        legacy="1.13",
    )
    np.format_float_scientific(1, precision=5)
    np.format_float_positional(1, trim="k")
    np.array_repr(AR)
    np.array_str(AR)

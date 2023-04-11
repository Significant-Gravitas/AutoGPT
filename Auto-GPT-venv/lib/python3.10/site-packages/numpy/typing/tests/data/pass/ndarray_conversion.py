import os
import tempfile

import numpy as np

nd = np.array([[1, 2], [3, 4]])
scalar_array = np.array(1)

# item
scalar_array.item()
nd.item(1)
nd.item(0, 1)
nd.item((0, 1))

# tolist is pretty simple

# itemset
scalar_array.itemset(3)
nd.itemset(3, 0)
nd.itemset((0, 0), 3)

# tobytes
nd.tobytes()
nd.tobytes("C")
nd.tobytes(None)

# tofile
if os.name != "nt":
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        nd.tofile(tmp.name)
        nd.tofile(tmp.name, "")
        nd.tofile(tmp.name, sep="")

        nd.tofile(tmp.name, "", "%s")
        nd.tofile(tmp.name, format="%s")

        nd.tofile(tmp)

# dump is pretty simple
# dumps is pretty simple

# astype
nd.astype("float")
nd.astype(float)

nd.astype(float, "K")
nd.astype(float, order="K")

nd.astype(float, "K", "unsafe")
nd.astype(float, casting="unsafe")

nd.astype(float, "K", "unsafe", True)
nd.astype(float, subok=True)

nd.astype(float, "K", "unsafe", True, True)
nd.astype(float, copy=True)

# byteswap
nd.byteswap()
nd.byteswap(True)

# copy
nd.copy()
nd.copy("C")

# view
nd.view()
nd.view(np.int64)
nd.view(dtype=np.int64)
nd.view(np.int64, np.matrix)
nd.view(type=np.matrix)

# getfield
complex_array = np.array([[1 + 1j, 0], [0, 1 - 1j]], dtype=np.complex128)

complex_array.getfield("float")
complex_array.getfield(float)

complex_array.getfield("float", 8)
complex_array.getfield(float, offset=8)

# setflags
nd.setflags()

nd.setflags(True)
nd.setflags(write=True)

nd.setflags(True, True)
nd.setflags(write=True, align=True)

nd.setflags(True, True, False)
nd.setflags(write=True, align=True, uic=False)

# fill is pretty simple

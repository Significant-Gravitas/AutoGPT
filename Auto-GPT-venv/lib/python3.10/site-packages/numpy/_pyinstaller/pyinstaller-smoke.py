"""A crude *bit of everything* smoke test to verify PyInstaller compatibility.

PyInstaller typically goes wrong by forgetting to package modules, extension
modules or shared libraries. This script should aim to touch as many of those
as possible in an attempt to trip a ModuleNotFoundError or a DLL load failure
due to an uncollected resource. Missing resources are unlikely to lead to
arithmetic errors so there's generally no need to verify any calculation's
output - merely that it made it to the end OK. This script should not
explicitly import any of numpy's submodules as that gives PyInstaller undue
hints that those submodules exist and should be collected (accessing implicitly
loaded submodules is OK).

"""
import numpy as np

a = np.arange(1., 10.).reshape((3, 3)) % 5
np.linalg.det(a)
a @ a
a @ a.T
np.linalg.inv(a)
np.sin(np.exp(a))
np.linalg.svd(a)
np.linalg.eigh(a)

np.unique(np.random.randint(0, 10, 100))
np.sort(np.random.uniform(0, 10, 100))

np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
np.ma.masked_array(np.arange(10), np.random.rand(10) < .5).sum()
np.polynomial.Legendre([7, 8, 9]).roots()

print("I made it!")

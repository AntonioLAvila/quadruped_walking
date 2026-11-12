"""Load torch's CUDA linalg library by absolute path. Import this before using mjlab.

torch keeps its GPU linear algebra in a separate ``libtorch_cuda_linalg.so`` and opens it
lazily, **by bare filename**, the first time a ``torch.linalg`` routine runs on a CUDA
tensor -- here, mjlab's inertia randomization and the terrain-normal fit behind the
``upright`` reward. A bare-filename ``dlopen`` is resolved against the search path of
whichever library the loader credits with the call. That is normally torch's own RPATH,
but when it is not, the lookup falls through to ``LD_LIBRARY_PATH`` (a sourced ROS
workspace, say), finds nothing, and training dies with "cannot open shared object file"
while the file sits in site-packages.

Opening the library once by its full path takes the search out of the picture: the loader
matches any later request against the SONAME of what is already mapped and never walks a
search path at all. It replaces the old ``bin/uv-run`` wrapper, which fixed the same thing
from outside by rewriting ``LD_LIBRARY_PATH`` and so only worked if you remembered to use
it.
"""

import ctypes
from pathlib import Path

import torch

_LIB = Path(torch.__file__).parent / "lib" / "libtorch_cuda_linalg.so"
if _LIB.exists():  # absent from CPU-only torch builds, which never open it
  ctypes.CDLL(str(_LIB))

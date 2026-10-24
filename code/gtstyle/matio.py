"""
gtstyle.matio -- loaders for the panel .mat data files.

Two .mat file formats occur in this dataset:
  * v7 and earlier  -> read with scipy.io.loadmat
  * v7.3 (HDF5)     -> read with h5py, applying the array transposition and
                       object-reference resolution that v7.3 requires

`loadmat` dispatches on the file magic number and returns a plain dict:

  * 1x1 structs become MatStruct objects with attribute access, so a field
    is reached as S.wt
  * char arrays become str
  * cell arrays become lists
  * singleton dimensions on numeric arrays are squeezed, so a stored column
    vector arrives as a flat 1-D array

Internal keys such as __function_workspace__ are dropped.
"""

import numpy as np
import scipy.io as sio


class MatStruct:
    """Attribute-accessible view of a 1x1 struct from a .mat file.

    Fields are reachable as both S.field and S['field'].
    """

    def __init__(self, d):
        self._d = dict(d)

    def __getattr__(self, k):
        try:
            return self._d[k]
        except KeyError as e:
            raise AttributeError(
                f"struct has no field {k!r}; available: {sorted(self._d)}"
            ) from e

    def __getitem__(self, k):
        return self._d[k]

    def __contains__(self, k):
        return k in self._d

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def get(self, k, default=None):
        return self._d.get(k, default)

    def __repr__(self):
        return f"MatStruct({sorted(self._d)})"


def _clean(x):
    """Recursively convert a scipy.io object into plain Python types."""
    # struct array
    if isinstance(x, np.ndarray) and x.dtype.names is not None:
        if x.size == 1:
            rec = x.ravel()[0]
            return MatStruct({n: _clean(rec[n]) for n in x.dtype.names})
        return [MatStruct({n: _clean(r[n]) for n in x.dtype.names})
                for r in x.ravel()]

    # cell array
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.size == 1:
            return _clean(x.ravel()[0])
        return [_clean(v) for v in x.ravel()]

    if isinstance(x, np.ndarray):
        # char array -> str
        if x.dtype.kind in ("U", "S"):
            flat = x.ravel()
            if flat.size == 1:
                return str(flat[0])
            return "".join(str(c) for c in flat)
        # numeric: drop singleton dimensions
        if x.ndim == 2 and 1 in x.shape:
            return x.ravel()
        if x.size == 1:
            return x.item()
        return x

    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf8", errors="replace")

    return x


def _load_v73(path):
    """Load a v7.3 (HDF5) .mat file into a dict."""
    import h5py

    def read(node, f):
        if isinstance(node, h5py.Group):
            return MatStruct({k: read(node[k], f) for k in node.keys()
                              if not k.startswith("#")})
        arr = node[()]
        # v7.3 stores char data as uint16 code points
        if node.attrs.get("MATLAB_class", b"") == b"char":
            return "".join(chr(c) for c in np.asarray(arr).ravel())
        # object references (cell arrays)
        if arr.dtype == h5py.ref_dtype:
            return [read(f[r], f) for r in np.asarray(arr).ravel()]
        arr = np.asarray(arr)
        # v7.3 stores arrays column-major, so they arrive transposed
        arr = arr.T
        if arr.ndim == 2 and 1 in arr.shape:
            return arr.ravel()
        if arr.size == 1:
            return arr.item()
        return arr

    with h5py.File(path, "r") as f:
        return {k: read(f[k], f) for k in f.keys() if not k.startswith("#")}


def loadmat(path):
    """Load a .mat file of either format version into a plain dict."""
    with open(path, "rb") as fh:
        magic = fh.read(128)

    if magic[:8] == b"\x89HDF\r\n\x1a\n" or b"MATLAB 7.3" in magic:
        return _load_v73(path)

    raw = sio.loadmat(path, squeeze_me=False, struct_as_record=True)
    return {k: _clean(v) for k, v in raw.items() if not k.startswith("__")}


def load_struct(path):
    """Load a .mat file as a MatStruct, so its top-level variables are
    reached by attribute: S.wt, S.sh, S.gt, S.ylab.
    """
    return MatStruct(loadmat(path))

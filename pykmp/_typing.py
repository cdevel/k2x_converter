from typing import Any, ClassVar, Generic, NewType, TypeVar, Union

import numpy as np
import numpy.typing as npt

_Element = TypeVar("_Element", bound=int)
DataClass = Union[type, Any]
Scalar = NewType("Scalar", int)
XYZ = NewType("XYZ", int)
XY = NewType("XY", int)
RGB = NewType("RGB", int)
LR = NewType("LR", int)
Group = NewType("Group", int)
Settings = NewType("Settings", int)
NScalar = NewType("NScalar", int)
NXYZ = NewType("NXYZ", int)
NProp = NewType("Nprop", int)

class _DTypeMeta(type):
    def __new__(cls, name, bases, attrs):
        assert "__type__" in attrs, f"__type__ not defined in {name}"
        return super().__new__(cls, name, bases, attrs)


class _Dtype(metaclass=_DTypeMeta):
    __type__: ClassVar[npt.DTypeLike] = None

    def __repr__(self):
        return self.__class__.__name__


class String(Generic[_Element], _Dtype):
    __type__ = str


class Bit(Generic[_Element], _Dtype):
    __type__ = np.dtype(np.bool_)


class Byte(Generic[_Element], _Dtype):
    __type__ = np.dtype(np.uint8)


class UInt16(Generic[_Element], _Dtype):
    __type__ = np.dtype(np.uint16)


class UInt32(Generic[_Element], _Dtype):
    __type__ = np.dtype(np.uint32)


class UInt64(Generic[_Element], _Dtype):
    __type__ = np.dtype(np.uint64)


class Int8(Generic[_Element], _Dtype):
    __type__ = np.dtype(np.int8)


class Int16(Generic[_Element], _Dtype):
    __type__ = np.dtype(np.int16)


class Int32(Generic[_Element], _Dtype):
    __type__ = np.dtype(np.int32)


class Int64(Generic[_Element], _Dtype):
    __type__ = np.dtype(np.int64)


class Float(Generic[_Element], _Dtype):
    __type__ = np.dtype(np.float32)


_supportdtype = [
    String,
    Bit,
    Byte, UInt16, UInt32, UInt64,
    Int8, Int16, Int32, Int64,
    Float
]
_argmap = {
    Scalar: 1, XYZ: 3, RGB: 3, LR: 2, XY: 2,
    Group: 6, Settings: 8,
    NXYZ: (None, 3), NScalar: (None, 1),  # used poti points
    NProp: (None, 4)
}


def get_colname(
    key: str,
    dtype: Any
) -> str | list[str]:
    if isinstance(dtype, np.dtype) or dtype in _supportdtype:
        return key
    elif (
        hasattr(dtype, '__origin__')
        and dtype.__origin__ in _supportdtype
        and hasattr(dtype.__origin__, '__type__')
    ):
        _elements = dtype.__args__[0]
        _counts = _argmap[_elements]
        if isinstance(_counts, tuple):
            _, _counts = _counts
        if _counts == 1:
            return key
        elif _elements.__name__ in ['XYZ', 'LR', 'XY', 'NXYZ', 'RGB']:
            name = _elements.__name__ if _elements.__name__ != 'NXYZ' else 'XYZ'
            return [key + '_' + k.lower() for k in name]
        return [key + '_' + str(i + 1) for i in range(_counts)]

    raise TypeError(f"Unsupported dtype: {dtype}")

def get_dtype_and_size(dtype: Any) -> tuple[np.dtype, int | tuple[None, int]]:
    """Get the dtype and elements size."""
    if isinstance(dtype, np.dtype):
        return dtype, 1
    elif dtype in _supportdtype:
        return dtype.__type__, 1
    elif (
        hasattr(dtype, '__origin__')
        and dtype.__origin__ in _supportdtype
        and hasattr(dtype.__origin__, '__type__')
    ):
        _dtype = dtype.__origin__.__type__
        _elements = dtype.__args__[0]
        if _elements in _argmap:
            _elements = _argmap[_elements]
        else:
            raise TypeError(f"Unsupported key: {_elements}")
        return _dtype, _elements

    raise TypeError(f"Unsupported dtype: {dtype}")

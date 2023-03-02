import dataclasses
import warnings
from types import EllipsisType
from typing import Iterable, Union

import numpy as np
from typing_extensions import Self

from pykmp._io._parser import _BinaryParser as Parser
from pykmp._typing import NXYZ, XYZ, Byte, Float, NScalar, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import CustomFnSpec, section_add_attrs
from pykmp.utils import tobytes


def _parse_poti_points(parser: Parser, numpoints: int):
    """Parse POTI Route points from POTI."""
    assert parser.is_read_contiuously

    pos = []
    property1 = []
    property2 = []

    for _ in range(numpoints):
        pos.append(parser.read_float32(3))
        property1.append(parser.read_uint16()[None])
        property2.append(parser.read_uint16()[None])

    return {
        'pos': np.array(pos, dtype=np.float32),
        'property1': np.array(property1, dtype=np.uint16),
        'property2': np.array(property2, dtype=np.uint16)
    }


POTI_SPEC = {
    'pos': CustomFnSpec(
        _parse_poti_points, ('pos', 'property1', 'property2'),
        additional_args=('numpoints',)
    )
}


@dataclasses.dataclass(eq=False)
class PotiPoints:
    index: Union[int, slice, Iterable[int], EllipsisType]
    pos: Float[XYZ]
    property1: UInt16
    property2: UInt16


@dataclasses.dataclass(eq=False)
class POTIStruct(BaseStruct):
    numpoints: UInt16
    smooth: Byte
    forward_backward: Byte
    pos: Float[NXYZ]
    property1: UInt16[NScalar]
    property2: UInt16[NScalar]

    def __len__(self: Self):
        return self.numpoints

    def tobytes(self: Self) -> bytes:
        b = tobytes(self.numpoints, self.numpoints.dtype)
        b += tobytes(self.smooth, self.smooth.dtype)
        b += tobytes(self.forward_backward, self.forward_backward.dtype)

        for i in range(self.numpoints):
            b += tobytes(self.pos[i])
            b += tobytes(self.property1[i])
            b += tobytes(self.property2[i])
        return b

    def __getitem__(self: Self, index: int | slice | Iterable[int]):
        if (isinstance(index, Iterable)
            and not all(isinstance(i, int) for i in index)):
            raise TypeError("Index must be int, slice, or iterable of ints")
        return PotiPoints(
            index,
            self.pos[index],
            self.property1[index],
            self.property2[index],
        )

    def __setitem__(
        self,
        index: int | slice | Iterable[int] | EllipsisType,
        value: PotiPoints
    ):
        if (
            isinstance(index, Iterable)
            and not all(isinstance(i, int) for i in index)
        ):
            raise TypeError("Index must be int, slice, or iterable of ints")
        self.pos[index] = value.pos
        self.property1[index] = value.property1
        self.property2[index] = value.property2


@section_add_attrs(POTIStruct, custom_fn=POTI_SPEC)
class POTI(BaseSection):
    numpoints: UInt16
    smooth: Byte
    forward_backward: Byte
    pos: Float[NXYZ]
    property1: UInt16[NScalar]
    property2: UInt16[NScalar]

    def _check_struct(self: Self, index: int, data: POTIStruct):
        if data.smooth > 1:
            warnings.warn(
                f"smooth of POTI #{index:X} must be 0 or 1, "
                "pykmp will set it to 0"
            )
            data.smooth = Byte.convert(0)
        if data.forward_backward > 1:
            warnings.warn(
                f"forward_backward of POTI #{index:X} must be 0 or 1, "
                "pykmp will set it to 0"
            )
            data.forward_backward = Byte.convert(0)

        if data.smooth == 1 and len(data) < 3:
            warnings.warn(
                f"smooth of POTI #{index:X} is 1, "
                "but the number of points is less than 3."
                "This will freeze the game."
            )

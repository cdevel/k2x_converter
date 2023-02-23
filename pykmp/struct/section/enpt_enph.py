import dataclasses
import warnings

import numpy as np

from pykmp._typing import XYZ, Byte, Float, Group, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs


@dataclasses.dataclass(eq=False)
class ENPTStruct(BaseStruct):
    pos: Float[XYZ]
    range: Float
    property1: UInt16
    property2: Byte
    property3: Byte


@section_add_attrs(ENPTStruct)
class ENPT(BaseSection):
    pos: Float[XYZ]
    range: Float
    property1: UInt16
    property2: Byte
    property3: Byte


@dataclasses.dataclass(eq=False)
class ENPHStruct(BaseStruct):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    dispatch1: Byte
    dispatch2: Byte


@section_add_attrs(ENPHStruct)
class ENPH(BaseSection):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    dispatch1: Byte
    dispatch2: Byte

    def _check_struct(self, index: int, data: ENPHStruct):
        super()._check_struct(index, data)
        if (
            np.all(data.prev == 255)
            and np.all(data.dispatch1 == 0)
            and np.all(data.dispatch2 == 0)
        ):
            warnings.warn(f"ENPH #{index:X} has no previous group.")
        if np.all(data.next == 255):
            warnings.warn(f"ENPH #{index:X} has no next group.")
        # dispatch1: 0 <= x <= 7.
        if np.any(data.dispatch1 > 7):
            print(data.dispatch1)
            raise ValueError(
                f"dispatch1(0x0E) must be 0 <= x <= 7. (ENPH #{index:X})")
        # https://wiki.tockdom.com/wiki/Enemy_routes_in_battle_arenas
        # dispatch2: 0x00, 0x40, 0x80, 0xC0.
        if np.any(data.dispatch2 & 0x3F):
            warnings.warn(
                "For dispatch2(0x0F), only 0x00, 0x40, 0x80, 0xC0 are used, "
                f"but ENPH #{index:X} has 0x{data.dispatch2:02X}."
            )

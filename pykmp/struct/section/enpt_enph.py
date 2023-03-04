import warnings

import numpy as np

from pykmp._typing import XYZ, Byte, Float, Group, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import (check_range, section_add_attrs,
                                         struct_decorate)


@struct_decorate
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


@struct_decorate(
    dispatch1=check_range(8),
)
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
        # 1. Every `next` must have at least one next whose value is not 255,
        # except for those for which dispatch is defined.
        if (
            np.all(data.prev == 255)
            and np.all(data.dispatch1 == 0)
            and np.all(data.dispatch2 == 0)
        ):
            warnings.warn(f"ENPH #{index:X} has no previous group.")
        # 2. Every `next` must have at least one next whose value is not 255.
        if np.all(data.next == 255):
            warnings.warn(f"ENPH #{index:X} has no next group.")
        # 3. If `dispath1` is defined, range must be 0 <= x <= 7.
        if np.any(data.dispatch1 > 7):
            raise ValueError(
                f"dispatch1(0x0E) must be 0 <= x <= 7. (ENPH #{index:X})")
        # 4. If `dispath2` is defined, value must be 0x00, 0x40, 0x80 or 0xC0.
        # see: https://wiki.tockdom.com/wiki/Enemy_routes_in_battle_arenas
        if np.any(data.dispatch2 & 0x3F):
            warnings.warn(
                "For dispatch2(0x0F), only 0x00, 0x40, 0x80, 0xC0 are used, "
                f"but ENPH #{index:X} has 0x{data.dispatch2:02X}."
            )

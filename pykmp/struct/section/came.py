from pykmp._typing import XYZ, Byte, Float, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import (check_range, section_add_attrs,
                                         struct_decorate)


@struct_decorate(
    type=check_range(10),
    time=check_range(6001) # 99:59.99
)
class CAMEStruct(BaseStruct):
    type: Byte
    next: Byte
    unknown: Byte
    route: Byte
    vcame: UInt16
    vzoom: UInt16
    vpt: UInt16
    unknown2: Byte
    unknown3: Byte
    pos: Float[XYZ]
    rot: Float[XYZ]
    zoombeg: Float
    zoomend: Float
    viewbegpos: Float[XYZ]
    viewendpos: Float[XYZ]
    time: Float


@section_add_attrs(CAMEStruct)
class CAME(BaseSection):
    type: Byte
    next: Byte
    unknown: Byte
    route: Byte
    vcame: UInt16
    vzoom: UInt16
    vpt: UInt16
    unknown2: Byte
    unknown3: Byte
    pos: Float[XYZ]
    rot: Float[XYZ]
    zoombeg: Float
    zoomend: Float
    viewbegpos: Float[XYZ]
    viewendpos: Float[XYZ]
    time: Float


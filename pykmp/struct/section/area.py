import dataclasses

from pykmp._typing import XYZ, Byte, Float, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs


@dataclasses.dataclass(eq=False)
class AREAStruct(BaseStruct):
    shape: Byte
    type: Byte
    cameindex: Byte
    priority: Byte
    pos: Float[XYZ]
    rot: Float[XYZ]
    scale: Float[XYZ]
    setting1: UInt16
    setting2: UInt16
    potiID: Byte
    enptID: Byte
    unknown: UInt16


@section_add_attrs(AREAStruct)
class AREA(BaseSection):
    shape: Byte
    type: Byte
    cameindex: Byte
    priority: Byte
    pos: Float[XYZ]
    rot: Float[XYZ]
    scale: Float[XYZ]
    setting1: UInt16
    setting2: UInt16
    potiID: Byte
    enptID: Byte
    unknown: UInt16

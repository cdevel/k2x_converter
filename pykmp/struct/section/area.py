import warnings

from pykmp._typing import XYZ, Byte, Float, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs, struct_decorate


@struct_decorate
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

    def _check_struct(self, index: int, data: AREAStruct):
        # 1. shape must 0 or 1.
        if data.shape > 1:
            warnings.warn(
                f"shape of AREA #{index:X} is out of range."
                f" value must be 0 or 1, but got {data.shape}."
            )

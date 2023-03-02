from pykmp._typing import XYZ, Float, Int16, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs, struct_decorate


@struct_decorate
class MSPTStruct(BaseStruct):
    pos: Float[XYZ]
    rot: Float[XYZ]
    entryID: UInt16
    unknown: Int16


@section_add_attrs(MSPTStruct)
class MSPT(BaseSection):
    pos: Float[XYZ]
    rot: Float[XYZ]
    entryID: UInt16
    unknown: Int16

import dataclasses

from pykmp._typing import XYZ, Float, Int16, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs


@dataclasses.dataclass(eq=False)
class CNPTStruct(BaseStruct):
    pos: Float[XYZ]
    rot: Float[XYZ]
    id: UInt16
    effect: Int16


@section_add_attrs(CNPTStruct)
class CNPT(BaseSection):
    pos: Float[XYZ]
    rot: Float[XYZ]
    id: UInt16
    effect: Int16

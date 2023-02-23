import dataclasses

from pykmp._typing import XYZ, Float, Int16, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs


@dataclasses.dataclass(eq=False)
class JGPTStruct(BaseStruct):
    pos: Float[XYZ]
    rot: Float[XYZ]
    unknown: UInt16
    range: Int16


@section_add_attrs(JGPTStruct)
class JGPT(BaseSection):
    pos: Float[XYZ]
    rot: Float[XYZ]
    unknown: UInt16
    range: Int16

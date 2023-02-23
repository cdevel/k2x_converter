import dataclasses

from pykmp._typing import XYZ, Byte, Float, Group, Int16, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs


@dataclasses.dataclass(eq=False)
class ITPTStruct(BaseStruct):
    pos: Float[XYZ]
    range: Float
    property1: UInt16
    property2: UInt16


@section_add_attrs(ITPTStruct)
class ITPT(BaseSection):
    pos: Float[XYZ]
    range: Float
    property1: UInt16
    property2: UInt16


@dataclasses.dataclass(eq=False)
class ITPHStruct(BaseStruct):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16


@section_add_attrs(ITPHStruct)
class ITPH(BaseSection):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16

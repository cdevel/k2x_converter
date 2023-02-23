import dataclasses

from pykmp._typing import XYZ, Float, Int16, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs


@dataclasses.dataclass(eq=False)
class KTPTStruct(BaseStruct):
    pos: Float[XYZ]
    rot: Float[XYZ]
    playerIndex: Int16
    unknown: UInt16


@section_add_attrs(KTPTStruct)
class KTPT(BaseSection):
    pos: Float[XYZ]
    rot: Float[XYZ]
    playerIndex: Int16
    unknown: UInt16

    def _check_struct(self, index: int, data: KTPTStruct):
        if data.playerIndex > 11:
            raise ValueError(
                f"playerIndex of KTPT #{index:X} is out of range. "
                f"expected x <= 11, got {data.playerIndex}."
            )

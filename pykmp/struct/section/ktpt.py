from typing_extensions import Self

from pykmp._typing import XYZ, Float, Int16, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import (check_range, section_add_attrs,
                                         struct_decorate)


@struct_decorate(playerIndex=check_range(12))
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

    def _check_struct(self: Self, index: int, data: KTPTStruct):
        # 1. playerIndex must 0 <= x <= 11.
        if data.playerIndex > 11:
            raise ValueError(
                f"playerIndex of KTPT #{index:X} is out of range. "
                f"expected x <= 11, got {data.playerIndex}. "
                "Ignore it if the battle supported by 24 players."
            )

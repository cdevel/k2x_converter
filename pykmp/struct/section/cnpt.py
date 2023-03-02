import warnings

from pykmp._typing import XYZ, Float, Int16, UInt16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs, struct_decorate


@struct_decorate
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

    def _check_struct(self, index: int, data: CNPTStruct):
        if data.effect > 2:
            warnings.warn(
                f"effect of CNPT #{index:X} is out of range; "
                f"Ignore it if additional type is defined in LEX. "
            )

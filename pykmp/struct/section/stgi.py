import dataclasses
import warnings

from typing_extensions import Self

from pykmp._io._parser import _BinaryParser as Parser
from pykmp._typing import RGB, Byte, Float
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import CustomFnSpec, section_add_attrs


def _parse_stgi_speedfactor(parser: Parser):
    """Parse STGI Speed factor from STGI for max speed modifier"""
    assert parser.is_read_contiuously

    unused2 = parser.read_uint8()
    speedfactor = parser.read_number('>f4', 1, size=2, fillbyte=b'\x00\x00')

    return {
        'unused2': unused2,
        'speedfactor': speedfactor
    }


STGI_SPEC = {
    'speedfactor': CustomFnSpec(
        _parse_stgi_speedfactor, ('unused2', 'speedfactor')
    )
}


@dataclasses.dataclass(eq=False)
class STGIStruct(BaseStruct):
    lap: Byte
    poleposition: Byte
    distancetype: Byte
    flareflash: Byte
    unused: Byte
    flarecolor: Byte[RGB]
    transparency: Byte
    unused2: Byte
    speedfactor: Float


@section_add_attrs(STGIStruct, indexing=False, custom_fn=STGI_SPEC)
class STGI(BaseSection):
    lap: Byte
    poleposition: Byte
    distancetype: Byte
    flareflash: Byte
    unused: Byte
    flarecolor: Byte[RGB]
    transparency: Byte
    unused2: Byte
    speedfactor: Float

    def tobytes(self: Self) -> bytes:
        return super().tobytes()[:-2]

    def _check_struct(self: Self, index: int, data: STGIStruct):
        super()._check_struct(index, data)
        if data.lap > 9:
            warnings.warn(
                f"lap must be less than 10. pykmp will set it to 9.")
            data.lap = Byte.convert(9)
        if data.poleposition > 1:
            warnings.warn(
                f"poleposition must be 0 or 1. pykmp will set it to 0.")
            data.poleposition = Byte.convert(0)
        if data.distancetype > 1:
            warnings.warn(
                f"distancetype must be 0 or 1. pykmp will set it to 0.")
            data.distancetype = Byte.convert(0)
        if data.flareflash > 1:
            warnings.warn(
                f"flareflash must be 0 or 1. pykmp will set it to 0.")
            data.flareflash = Byte.convert(0)

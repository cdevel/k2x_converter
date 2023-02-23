import dataclasses

import numpy as np

from pykmp._typing import XY, Byte, Float, Group, Int16
from pykmp.struct.core import BaseSection, BaseStruct
from pykmp.struct.section._utils import section_add_attrs


@dataclasses.dataclass(eq=False)
class CKPTStruct(BaseStruct):
    left: Float[XY]
    right: Float[XY]
    respawn: Byte
    mode: Byte
    prev: Byte
    next: Byte


@section_add_attrs(CKPTStruct)
class CKPT(BaseSection):
    left: Float[XY]
    right: Float[XY]
    respawn: Byte
    mode: Byte
    prev: Byte
    next: Byte


@dataclasses.dataclass(eq=False)
class CKPHStruct(BaseStruct):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16


@section_add_attrs(CKPHStruct)
class CKPH(BaseSection):
    start: Byte
    length: Byte
    prev: Byte[Group]
    next: Byte[Group]
    unknown: Int16


def fix_pt_prev_next(ckpt: CKPT, ckph: CKPH):
    """
    Fix the prev and next of the checkpoint (CKPT).
    Note this function will modify the `ckpt` in-place.

    Args:
        ckpt (CKPT): The checkpoint. must have linked CKPH.
        ckph (CKPH): The checkpoint path. must have linked CKPT.
    """
    new_p_n: np.ndarray = None
    for start, length in zip(ckph.start, ckph.length):
        arange = np.arange(start, start + length)
        p_n = np.r_[
            '1,2,0',
            np.r_[255, arange[:-1]], np.r_[arange[1:], 255]
        ]
        if new_p_n is None:
            new_p_n = p_n
        else:
            new_p_n = np.vstack((new_p_n, p_n))
    ckpt.prev, ckpt.next = map(lambda x: x.flatten(), np.hsplit(new_p_n, 2))

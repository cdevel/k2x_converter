from pykmp.struct.section.area import AREA as AREA
from pykmp.struct.section.came import CAME as CAME
from pykmp.struct.section.ckpt_ckph import CKPH as CKPH
from pykmp.struct.section.ckpt_ckph import CKPT as CKPT
from pykmp.struct.section.ckpt_ckph import fix_pt_prev_next as fix_pt_prev_next
from pykmp.struct.section.cnpt import CNPT as CNPT
from pykmp.struct.section.enpt_enph import ENPH as ENPH
from pykmp.struct.section.enpt_enph import ENPT as ENPT
from pykmp.struct.section.gobj import GOBJ as GOBJ
from pykmp.struct.section.itpt_itph import ITPH as ITPH
from pykmp.struct.section.itpt_itph import ITPT as ITPT
from pykmp.struct.section.jgpt import JGPT as JGPT
from pykmp.struct.section.ktpt import KTPT as KTPT
from pykmp.struct.section.mspt import MSPT as MSPT
from pykmp.struct.section.poti import POTI as POTI
from pykmp.struct.section.stgi import STGI as STGI

_KMP_CLASSES = [
    KTPT,
    ENPT, ENPH,
    ITPT, ITPH,
    CKPT, CKPH,
    GOBJ,
    POTI,
    AREA,
    CAME,
    JGPT,
    CNPT,
    MSPT,
    STGI
]
_KMP_NAME_AND_CLASSES = [(cls.__rname__, cls) for cls in _KMP_CLASSES]


def section_classes(type_: str = 'kmp') -> list[tuple[str, type]]:
    if type_ == 'kmp':
        return _KMP_NAME_AND_CLASSES
    raise NotImplementedError('Currently only kmp is supported.')

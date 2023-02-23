import dataclasses
import itertools
import os
import pathlib
import re
import string
import warnings

import numpy as np
import pandas as pd
from typing_extensions import Self

#from pykmp import wiimm
from pykmp._io import _parser
from pykmp.struct import section
from pykmp.struct.pandas_utils import merge_pt_ph, split_pt_ph
from pykmp.utils import tobytes

_RKMD_STRUCT = (
    'KTPT', 'ENPT', 'ENPH', 'ITPT', 'ITPH', 'CKPT', 'CKPH', 'GOBJ', 'POTI',
    'AREA', 'CAME', 'JGPT', 'CNPT', 'MSPT', 'STGI',# 'WIM0'
)


@dataclasses.dataclass(eq=False)
class KMP:
    """KMP file class"""
    header: str
    version_number: np.uint32
    KTPT: section.KTPT
    ENPT: section.ENPT
    ENPH: section.ENPH
    ITPT: section.ITPT
    ITPH: section.ITPH
    CKPT: section.CKPT
    CKPH: section.CKPH
    GOBJ: section.GOBJ
    POTI: section.POTI
    AREA: section.AREA
    CAME: section.CAME
    JGPT: section.JGPT
    CNPT: section.CNPT
    MSPT: section.MSPT
    STGI: section.STGI
    # WIM0: section.WIM0

    def __post_init__(self):
        section.fix_pt_prev_next(self.CKPT, self.CKPH)

    def __repr__(self: Self):
        _repr = '<class \'pykmp.struct.kmp.KMP\'>\n'
        _repr += 'header: {}\n'.format(self.header)
        _repr += 'version_number: {:X}\n'.format(self.version_number)
        for name, _ in section.section_classes('kmp'):
            cls_ = getattr(self, name)
            _repr += '{}: {:3d} entries'.format(name, cls_.entries)
            if name == 'GOBJ' and cls_.le_mode:
                _repr += ' (has LE-CODE mode)'
            _repr += '\n'
        return _repr[:-1]

    def __eq__(self: Self, other: Self) -> bool:
        cond = self.header == other.header
        cond &= self.version_number == other.version_number
        for name, _ in section.section_classes('kmp'):
            if not __debug__:
                print(name, getattr(self, name) == getattr(other, name))
            cond &= getattr(self, name) == getattr(other, name)
        return cond

    @property
    def le_mode(self: Self) -> bool:
        return self.GOBJ.le_mode

    def write(self: Self, path: str):
        # collect section bytes
        sectiondata = b""
        section_size = []
        for name, _ in section.section_classes('kmp'):
            data = getattr(self, name).tobytes()
            sectiondata += data
            section_size.append(len(data))

        section_offsets = [0] + list(itertools.accumulate(section_size))[:-1]

        header_length = 0x4C
        # for future implementation (e.g. WIM0 support)
        if len(section_offsets) > 15:
            header_length += 4 * (len(section_offsets) - 15)
        byte_length = len(sectiondata) + header_length

        # write headers
        # header
        b = self.header.encode('utf-8')
        # byte length
        b += tobytes(int(byte_length), np.uint32)
        # total sections
        b += tobytes(len(section_size), np.uint16)
        # header length
        b += tobytes(int(header_length), np.uint16)
        # version number
        b += tobytes(int(self.version_number), np.uint32)
        # section offsets
        b += tobytes(section_offsets, np.uint32)

        with open(path, 'wb') as f:
            f.write(b)
            f.write(sectiondata)

    def to_excel(self, path: str, hex_mode: bool = True, **kwargs):
        if 'mode' in kwargs:
            raise ValueError('mode is not allowed.')

        data = {}
        for name, _ in section.section_classes('kmp'):
            df = getattr(self, name).to_dataframe(hex_mode)
            data[name] = df
            if name in ['ENPT', 'ITPT', 'CKPT']:
                head = name[:-2]
                data[f'{head}PT+{head}PH'] = None

        # merge
        data['ENPT+ENPH'] = merge_pt_ph(data['ENPT'], data['ENPH'])
        data['ITPT+ITPH'] = merge_pt_ph(data['ITPT'], data['ITPH'])
        data['CKPT+CKPH'] = merge_pt_ph(data['CKPT'], data['CKPH'])
        data.pop('ENPT'); data.pop('ENPH')
        data.pop('ITPT'); data.pop('ITPH')
        data.pop('CKPT'); data.pop('CKPH')

        _COLS = list(string.ascii_uppercase)
        _COLS += list(map(lambda x: 'A' + x, _COLS))

        with pd.ExcelWriter(path, mode='w', **kwargs) as writer:
            for name, df in data.items():
                df.to_excel(writer, sheet_name=name, index=False)
                for i, col in enumerate(df.columns):
                    if len(col) <= 7:
                        continue
                    writer.book[name].column_dimensions[
                        _COLS[i]].width = len(col) + 3

    @classmethod
    def from_excel(cls, path: str):
        sheet_names = []
        section_clss = section.section_classes('kmp')
        for name, _ in section_clss:
            if name[:2] in ['EN', 'IT', 'CK']:
                head = name[:-2]
                sheet_names.append(f'{head}PT+{head}PH')
            else:
                sheet_names.append(name)

        init_kwgs = {
            'header': 'RKMD',
            'version_number': np.uint32(0x9D8),
        }
        data = pd.read_excel(path, sheet_name=sheet_names)
        section_clss = dict(section_clss)
        for key, df in data.items():
            match_res = re.match(r'(?P<pt>\w+PT)\+(?P<ph>\w+PH)', key)
            if match_res is not None:
                key_pt = match_res.group('pt')
                key_ph = match_res.group('ph')
                df_pt, df_ph = split_pt_ph(df, key_ph.upper() + ' ID')
                init_kwgs[key_pt] = section_clss[key_pt](df_pt)
                ph = section_clss[key_ph](df_ph)
                roll = np.roll(ph.start, -1) - ph.start
                roll[-1] = len(init_kwgs[key_pt]) - ph.start[-1]
                ph.length = roll.astype(np.uint8)
                init_kwgs[key_ph] = ph
            else:
                init_kwgs[key] = section_clss[key](df)

        return cls(**init_kwgs)


def read_kmp(
    path: str | pathlib.Path | os.PathLike,
):
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    parser = _parser._BinaryParser(path.read_bytes())

    assert parser.header == 'RKMD', (
        'Only Mario Kart Wii KMP formats are supported. '
        f'Expected RKMD, got {parser.header}.'
    )

    kwgs = {}
    kwgs['header'] = parser.header

    with parser.read_contiuously(start=0x04, back=True):
        _ = parser.read_uint32()
        total_sections = parser.read_uint16()
        header_length = parser.read_uint16()
        version_number = parser.read_uint32()
        if version_number != 0x9D8:
            warnings.warn(
                'Different version number may cause unexpected behavior '
                'or raise errors. '
            )
        kwgs['version_number'] = version_number
        section_offsets = parser.read_uint32(int(total_sections))

    section_cls = section.section_classes('kmp')
    section_offsets = section_offsets[:len(section_cls)]
    for i, (name, cls) in enumerate(section_cls):
        if not __debug__:
            print(f'Parsing {name}...')
        kwgs[name] = cls(
            parser, offset=int(header_length + section_offsets[i])
        )

    del parser
    return KMP(**kwgs)

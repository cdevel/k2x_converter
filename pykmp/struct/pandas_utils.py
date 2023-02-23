import warnings
from collections import defaultdict
from itertools import groupby
from typing import TypeVar

import numpy as np
import pandas as pd

from pykmp import _typing as t
from pykmp.struct.descriptor import DataDescriptor, _ListMax255, _SpecialName

Section = TypeVar('Section')
Struct = TypeVar('Struct')


def merge_pt_ph(pt: pd.DataFrame, ph: pd.DataFrame) -> pd.DataFrame:
    cols = list(pt.columns) + list(ph.columns)
    df = pd.DataFrame(columns=cols)

    for i in range(ph.shape[0]):
        start = ph.loc[i, 'start']
        if not isinstance(start, int):
            start = int(str(start), 0)
        length = ph.loc[i, 'length']
        if not isinstance(length, int):
            length = int(str(length), 0)
        loc_pt = pt.loc[start:start + length - 1]
        loc_pt = loc_pt.reset_index(drop=True)
        loc_ph = ph.iloc[i:i+1, :]
        # add nan rows
        nan_df = pd.concat([loc_ph] * (length - 1), ignore_index=True)
        # all values are NaN
        for col in nan_df.columns:
            nan_df[col] = np.nan

        loc_ph = pd.concat([loc_ph, nan_df], ignore_index=True, axis=0)
        df_ = pd.concat([loc_pt, loc_ph], axis=1)
        df = pd.concat([df, df_], ignore_index=True, axis=0)

    return df


def split_pt_ph(
    df: pd.DataFrame, value_if_missing: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(df.columns)
    try:
        start_idx = cols.index('start')
    except ValueError as e:
        if value_if_missing is None:
            raise e
        # compatible with old version
        try:
            start_idx = cols.index(value_if_missing)
        except ValueError:
            raise e
    df_pt = df.iloc[:, :start_idx]
    df_ph = df.iloc[:, start_idx:].dropna(axis=0, how='all')
    if df_ph.columns[0] != value_if_missing:
        df_ph = df_ph.reset_index(drop=True)
    else:
        # to detect old version
        df_ph.insert(0, "Unnamed: 0", np.nan)
    return df_pt, df_ph


def to_dataframe(
    section: Section,
    hex_mode: bool = False,
    no_nan: bool = False
) -> pd.DataFrame:
    cols = []
    name = section.__rname__
    if name == 'POTI':
        cols.append('index')
    elif name == 'CAME':
        cols.append('op_camera')
    for k, dt in section.__annotations__.items():
        colname = t.get_colname(k, dt)
        if isinstance(colname, list):
            cols.extend(colname)
        else:
            cols.append(colname)

    # create the data
    data = defaultdict(list)
    toT = lambda x: x.T if isinstance(x, np.ndarray) else x
    for index, rd in enumerate(section._rdata):
        vals = rd.tolist(hex_mode)
        appender = 'append'
        if name == 'POTI':
            *rdl, = map(toT, [index] + vals)
            numpts = rd.numpoints
            # convert all elements to same size
            vals = []
            for val in rdl:
                if isinstance(val, np.ndarray):
                    vals.extend(val.tolist())
                else:
                    fillv = val if no_nan else np.nan
                    vals.append([val] + [fillv] * (numpts - 1))
            appender = 'extend'
        elif name == 'CAME':
            vals = [1 if section.op_camera == index else np.nan] + vals
        for col, val in zip(cols, vals):
            getattr(data[col], appender)(val)

    df = pd.DataFrame(data, columns=cols)
    if name == 'POTI':
        df.drop(columns='numpoints', inplace=True)
    return df


def _gby_key(name):
    head = name.split('_')[0]
    if head == 'pf' or head == 'defobj' or head == 'lecode':
        return name
    return head


def _assert_size_mismatch(pred, actual):
    assert pred == actual, (
        "The number of columns in the dataframe is not correct. "
        "Expected {}, got {}.".format(pred, actual)
    )


def _fix_dataframe(df: pd.DataFrame, section: Section):
    """Fix the dataframe if it is from an old version (0.1)"""
    if df.columns[0] == "Unnamed: 0":
        # start to fix
        renamed_cols = []
        for key, type_ in section.__annotations__.items():
            cols = t.get_colname(key, type_)
            if isinstance(cols, list):
                renamed_cols.extend(cols)
            else:
                renamed_cols.append(cols)

        df.drop(columns="Unnamed: 0", inplace=True)

        name = section.__rname__
        # For developers:
        # All names substituted below are appropriate.
        # Don't worry about the names,
        # as they will all be replaced by `df.rename`.
        if name == 'KTPT' or name == 'AREA' or name == 'MSPT':
            # old version has no 'unknown' column
            _assert_size_mismatch(len(renamed_cols) - 1, len(df.columns))
            df.rename(
                columns=dict(zip(df.columns, renamed_cols[:-1])),
                inplace=True
            )
            df = df.assign(unknown=0)
        elif name == 'ENPH' or name == 'ITPH' or name == 'CKPH':
            if name != 'ENPH':
                df = df.assign(unknown=0)
            df.drop(columns=f"{name} ID", inplace=True)
            df.insert(0, 'start', df.index)
            df.insert(1, 'length', df.index)
            _assert_size_mismatch(len(renamed_cols), len(df.columns))
            df.rename(
                columns=dict(zip(df.columns, renamed_cols)),
                inplace=True
            )
        elif name == 'GOBJ':
            df.insert(2, 'preserved', 0)
            df.insert(25, 'unused', 0)
            # Reference (hex) -> hex to int
            refid = df['Reference (hex)'].copy().apply(lambda x: int(x, 0))
            df.drop(columns='Reference (hex)', inplace=True)
            df.insert(4, 'Reference (hex)', refid)
            _assert_size_mismatch(len(renamed_cols), len(df.columns))
            df.rename(
                columns=dict(zip(df.columns, renamed_cols)),
                inplace=True
            )
        elif name == 'CAME':
            renamed_cols = ['op_camera'] + renamed_cols
            fcamera = df.pop('First1')
            _ = df.pop('First2')
            df.insert(0, 'op_camera', fcamera)
            df.insert(7, 'unknown2', 0)
            df.insert(8, 'unknown3', 0)
            df.insert(3, 'unknown', 0)
            _assert_size_mismatch(len(renamed_cols), len(df.columns))
            df.rename(
                columns=dict(zip(df.columns, renamed_cols)),
                inplace=True
            )
        elif name == 'STGI':
            df.insert(4, 'unused', 0)
            df.insert(9, 'unused2', 0)
            _assert_size_mismatch(len(renamed_cols), len(df.columns))
            df.rename(
                columns=dict(zip(df.columns, renamed_cols)),
                inplace=True
            )
        elif name == 'CKPT':
            df = df.assign(prev=0)
            df = df.assign(next=0)
            _assert_size_mismatch(len(renamed_cols), len(df.columns))
            df.rename(
                columns=dict(zip(df.columns, renamed_cols)),
                inplace=True
            )
        elif name == 'JGPT':
            df.insert(6, 'unknown', 0)
            _assert_size_mismatch(len(renamed_cols), len(df.columns))
            df.rename(
                columns=dict(zip(df.columns, renamed_cols)),
                inplace=True
            )
        else:
            _assert_size_mismatch(len(renamed_cols), len(df.columns))
            df.rename(
                columns=dict(zip(df.columns, renamed_cols)),
                inplace=True
            )
    return df


def _add_0x(value):
    if isinstance(value, str):
        if value[:2] != '0x':
            value = '0x' + value
        return value
    # pd.Series
    return value.map(lambda x: '0x' + str(x))


def _convert_if_hex(value, dtype_hint: np.dtype):
    if not dtype_hint.kind == 'u':
        return value

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return int(_add_0x(value), 0)
    elif isinstance(value, pd.Series):
        try:
            return value.map(int)
        except ValueError:
            return value.map(lambda x: int(_add_0x(x), 0))
    return value


def from_dataframe(
    df: pd.DataFrame,
    section: Section,
) -> None:
    init_kwgs = {}
    init_kwgs['section'] = section.__rname__
    df = _fix_dataframe(df.copy(), section)
    name = section.__rname__

    if name == 'POTI':
        init_kwgs['entries'] = np.uint16(len(df.dropna(axis=0)))
    else:
        if name == 'STGI' and len(df) > 1:
            raise ValueError(
                f'Section STGI must have only one entry. got {len(df)}.'
            )
        init_kwgs['entries'] = np.uint16(len(df))

    special_name = _SpecialName.get(name, 'ignored')
    iloc_index = list(range(init_kwgs['entries']))

    # analyze section header
    if name == 'CAME':
        try:
            op_camera = int(df.query('op_camera > 0')['op_camera'].index[0])
        except (ValueError, TypeError) as e:
            raise ValueError(
                'Cannot find the opening camera. See the error below.') from e
        if op_camera not in list(range(init_kwgs['entries'])):
            warnings.warn(
                'The CAME opening index is not in the itemkey. '
                'This may cause freezing in the game.',
            )
        init_kwgs['additional'] = np.uint8(op_camera)
        init_kwgs['padding'] = np.uint8(0)
        init_kwgs['special_name'] = special_name
        df.drop(columns='op_camera', inplace=True)
    elif name == 'POTI':
        init_kwgs['additional'] = np.uint16(len(df))
        init_kwgs['padding'] = None
        init_kwgs['special_name'] = special_name
        current = 0
        iloc_index = defaultdict(list)
        try:
            index = df['index']
        except KeyError:
            df = df.rename(columns={'numpoints': 'index'})
            index = df['index']
        for i, idx in enumerate(index):
            if not np.isnan(idx):
                current = int(idx)
            iloc_index[current].append(i)
        df.drop(columns='index', inplace=True)
    else:
        init_kwgs['additional'] = np.uint8(0)
        init_kwgs['padding'] = None
        init_kwgs['special_name'] = None

    # setup attibute names
    attr_names = []
    colindex = 0
    for k, v in groupby(df.columns, key=_gby_key):
        v = list(v)
        if len(v) == 1:
            attr_names.append((v[0], colindex))
            colindex += 1
        else:
            attr_names.append((k, slice(colindex, colindex + len(v))))
            colindex += len(v)

    del colindex

    _rdata = _ListMax255()
    annotations = section.__annotations__

    # create rdata
    for i in range(init_kwgs['entries']):
        _init_kwgs = {}
        if section.__rname__ == 'POTI':
            loc_df = df.iloc[iloc_index[i]]
            _init_kwgs['numpoints'] = np.uint16(len(loc_df))
        else:
            loc_df = df.iloc[i]
        for j, (attr_name, cols) in enumerate(attr_names):
            dt, size = t.get_dtype_and_size(annotations[attr_name])
            if section.__rname__ == 'POTI' and j < 2:
                data = loc_df.iloc[0, cols]
            elif isinstance(loc_df, pd.DataFrame):
                data = loc_df.iloc[:, cols]
            else:
                data = loc_df.iloc[cols]
            # convert hex string to int(if)
            data = dt.type(_convert_if_hex(data, dt))
            if isinstance(size, tuple):
                data = data.reshape(-1, size[1])
            _init_kwgs[attr_name] = data
        _rdata.append(section.__struct__(**_init_kwgs))
    init_kwgs['descriptor'] = _rdata
    descriptor = DataDescriptor(**init_kwgs)
    section._init_from_descriptor(descriptor)

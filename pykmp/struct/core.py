import dataclasses
import warnings
from struct import pack as fpack
from types import EllipsisType
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

np.set_printoptions(precision=3, suppress=True)

from typing_extensions import Self

import pykmp._typing as t
from pykmp._io._parser import _BinaryParser as Parser
from pykmp.struct import pandas_utils
from pykmp.struct.descriptor import DataDescriptor, _ListMax255, _SpecialName
from pykmp.utils import tobytes


class HexPrinter:
    def __init__(self, cls: Union["BaseSection", "BaseStruct"]):
        self._cls = cls.copy()

    def __getattr__(self, name: str) -> Any:
        x = getattr(self._cls, name)
        if isinstance(x, np.ndarray) and x.dtype.kind in 'u?':
            return np.vectorize(hex)(x)
        elif isinstance(x, (float, np.float32)):
            return hex(int.from_bytes(fpack('>f', x), 'big'))
        elif isinstance(x, np.ndarray) and x.dtype.kind == 'f':
            return np.vectorize(
                lambda x: hex(int.from_bytes(fpack('>f', x), 'big'))
            )(x)
        try:
            return hex(x)
        except:
            return x


class BaseStruct:
    """Base class for all KMP structs"""
    def __eq__(self: Self, other: Self) -> bool:
        _data = dataclasses.asdict(self)
        _other_data = dataclasses.asdict(other)
        return all(np.array_equal(_data[k], _other_data[k]) for k in _data)

    def __copy__(self: Self) -> Self:
        _data = {k: v.copy() for k, v in dataclasses.asdict(self).items()}
        return self.__class__(**_data)

    def __setattr__(self: Self, __name: str, __value: Any) -> None:
        if hasattr(self, __name):
            _tvalue = getattr(self, __name)
            __value = np.array(__value, dtype=_tvalue.dtype)
            if __value.shape != _tvalue.shape:
                raise ValueError(
                    f"Cannot change the shape of {__name} to {__value.shape}"
                )
        super().__setattr__(__name, __value)

    def tolist(self: Self, as_hex: bool = False) -> dict[str, Any]:
        ret = []
        for v in dataclasses.asdict(self).values():
            if v.shape == ():  # scalar
                item = v.item()
                if as_hex and v.dtype.kind == 'u':
                    item = hex(item)
                ret.append(item)
                continue

            if as_hex and v.dtype.kind == 'u':
                v = np.vectorize(hex)(v)

            if len(v.shape) == 1:
                appender = 'extend'
                v = v.tolist()
            else: # POTI
                appender = 'append'
            getattr(ret, appender)(v)
        return ret

    def tobytes(self: Self) -> bytes:
        _bytes = []
        for v in dataclasses.asdict(self).values():
            _bytes.append(tobytes(v, v.dtype))
        return b''.join(_bytes)

    def check(self, raises: bool = True, fix_if_possible: bool = False):
        pass

    @property
    def hex(self: Self):
        """Use this method to print values as hex"""
        return HexPrinter(self)

    def copy(self: Self) -> Self:
        """Copy the struct."""
        return self.__copy__()


class BaseSection:
    """Base class for all KMP/LEX sections"""
    def __init_subclass__(cls, *args, **kwargs) -> None:
        annotations = cls.__annotations__
        if not annotations:
            raise TypeError(
                "Annotations shoud be defined "
                f"when subclassing {cls.__name__}"
            )
        cls._metadata = {}
        for k, value in annotations.items():
            try:
                dt, v = t.get_dtype_and_size(value)
            except TypeError:
                raise TypeError(f"Unsupported type {value}") from None
            else:
                if isinstance(value, str):
                    raise TypeError("String is not supported")
            cls._metadata[k] = (dt, v)
        super().__init_subclass__(*args, **kwargs)

    def __init__(
        self: Self,
        obj: Parser | DataDescriptor | pd.DataFrame,
        offset: int | None = None
    ) -> None:
        """
        Create a new instance of the structure

        Args:
            obj (_BinaryParser, DataDescriptor, pd.DataFrame): Input object
            offset (int, optional): Offset of the section in the file. Defaults to None.
        """
        if self.__class__.__name__ == "BaseSection":
            raise TypeError("BaseSection cannot be instantiated")

        if isinstance(obj, Parser):
            if offset is None:
                raise TypeError("Offset should be provided when using parser")
            with obj.read_contiuously(offset, back=True):
                self._init_from_parser(obj)
        elif isinstance(obj, DataDescriptor):
            self._init_from_descriptor(obj)
        elif isinstance(obj, pd.DataFrame):
            self._init_from_dataframe(obj)
        else:
            raise TypeError(
                "Input should be either a Parser or a DataDescriptor instance. "
                f"Got {type(obj)}"
            )
        self._check_section()

    def __getitem__(
        self: Self,
        itemkey: int | slice | Sequence[int] | EllipsisType
    ) -> Self:
        assert self.__indexing__, f"{self.section} is not indexing"
        if isinstance(itemkey, int):
            return self._rdata[itemkey]
        elif isinstance(itemkey, (slice, tuple, list, np.ndarray)):
            return self.__class__(self._to_descriptor(itemkey))
        elif isinstance(itemkey, EllipsisType):
            return self.copy()
        raise TypeError(f"Unsupported type {type(itemkey)}")

    def __getattr__(self: Self, __name: str):
        if __name in self._metadata:
            return self._pgetter(__name)
        try:
            return super().__getattr__(__name)
        except AttributeError:
            # Default error message is
            # "'super' object has no attribute '__getattr__'.",
            # which is not helpful.
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute "
                f"'{__name}'"
            ) from None

    def __setattr__(self: Self, __name: str, __value: Any) -> None:
        if __name in self._metadata:
            return self._psetter(__name, __value)
        super().__setattr__(__name, __value)

    def __len__(self: Self):
        return self.entries

    def __copy__(self: Self) -> Self:
        return self.__class__(self._to_descriptor(None, copy=True))

    def __eq__(self: Self, other: Self):
        if not isinstance(other, self.__class__):
            return False
        return self._to_descriptor(None) == other._to_descriptor(None)

    def __dataframe__(self: Self, **kwargs: Any) -> pd.DataFrame:
        return self.to_dataframe()

    def _repr_html_(self: Self) -> str:
        return self.to_dataframe()._repr_html_()

    def add(
        self: Self, obj: Self | BaseStruct, copy: bool = True
    ) -> Self:
        """Add a new struct/section to the section"""
        assert self.__indexing__, f"Cannot add to {self.section}"
        if dataclasses.is_dataclass(obj):
            if self._rdata[0] != obj:
                raise ValueError("Cannot append different Struct.")
            self._rdata.append(obj)
        elif isinstance(obj, self.__class__):
            if self._rdata[0] != obj._rdata[0]:
                raise ValueError("Cannot add different Section.")
            self._rdata.extend(obj._rdata)
        else:
            raise TypeError(
                f"Cannot add {obj.__class__.__name__} to "
                f"{self.section}."
            )
        self._sync_entries()
        if copy:
            return self.copy()
        return self

    def copy(self: Self) -> Self:
        """Copy the section."""
        return self.__copy__()

    @property
    def metadata(self: Self) -> dict[str, int | tuple[None, int]]:
        """Return the metadata of the section"""
        return self._metadata

    @property
    def hex(self: Self) -> HexPrinter:
        return HexPrinter(self)

    def _pgetter(self: Self, name: str) -> np.ndarray:
        if not self.__indexing__:
            return getattr(self._rdata[0], name)
        # gather values from self._rdata
        arrdata = [getattr(rd, name) for rd in self._rdata]
        if not all(arrdata[0].shape == arr.shape for arr in arrdata[1:]):
            # poti
            arry = np.concatenate(arrdata, axis=0)
        else:
            arry = np.array(arrdata, dtype=self._metadata[name][0])
        return arry

    def _psetter(self: Self, name: str, value: np.ndarray):
        dt, elem = self._metadata[name]
        dt, _ = t.get_dtype_and_size(dt)
        value = dt.type(value)
        if not self.__indexing__ and value.ndim == 0:
            value = value[None]
        if isinstance(elem, int):
            elem = (elem,)
        expected_shape = (int(self.entries), *elem)
        # TODO: supprt single value
        if len(expected_shape) != value.ndim:
            if not (
                len(expected_shape) > 1
                and expected_shape[-1] == 1
                and value.ndim == len(expected_shape) - 1
            ):
                raise ValueError(
                    f"Shape mismatch. Expected {expected_shape}, "
                    f"got {value.shape}"
                )
        for i, rd in enumerate(self._rdata):
            setattr(rd, name, value[i])

    def _to_descriptor(
        self: Self,
        itemkey : slice | Sequence[int] | None,
        copy: bool = False
    ) -> DataDescriptor:
        kwg = dict()
        kwg['section'] = self.section

        if isinstance(itemkey, slice):
            itemkeys = list(range(*itemkey.indices(self.entries)))
            kwg['entries'] = np.uint16(len(itemkeys))
        else:
            if isinstance(itemkey, np.ndarray):
                itemkey = itemkey.tolist()
            itemkeys = list(itemkey or range(self.entries))
            kwg['entries'] = np.uint16(len(itemkeys))

        special_name = _SpecialName.get(self.section, 'ignored')
        if self.section == 'CAME':
            op_camera = getattr(self, special_name[0])
            if op_camera not in itemkeys:
                warnings.warn(
                    'The CAME opening index is not in the itemkey. '
                    'This may cause freezing in the game.',
                )
            padding = getattr(self, special_name[1])
            kwg['additional'] = op_camera
            kwg['padding'] = padding
            kwg['special_name'] = special_name
        elif special_name == 'total_points':
            kwg['additional'] = np.uint16(sum(a.numpoints for a in self._rdata))
            kwg['padding'] = None
            kwg['special_name'] = special_name
        else:
            kwg['additional'] = np.uint16(getattr(self, special_name))
            kwg['padding'] = None
            kwg['special_name'] = None
        kwg['descriptor'] = _ListMax255()

        for i in itemkeys:
            rd = self._rdata[i]
            if copy:
                rd = rd.copy()
            kwg['descriptor'].append(rd)

        return DataDescriptor(**kwg)

    def tobytes(self: Self):
        """Convert section to bytes. Used for writing to file."""
        descriptor = self._to_descriptor(None)
        b = b''
        b += tobytes(descriptor.section)
        #b += desciptor.section.encode('utf-8')
        b += tobytes(int(descriptor.entries), np.uint16)

        b += tobytes(descriptor.additional)
        if descriptor.padding is not None:
            b += tobytes(descriptor.padding)
        b += b''.join(rd.tobytes() for rd in descriptor.descriptor)
        return b

    def to_dataframe(self: Self, hex_mode: bool = False, no_nan: bool = False):
        """
        Convert section data to a pandas DataFrame.

        Args:
            no_nan (bool, optional): If True, NaN values will be replaced with
            hex_mode (bool, optional): If True, unsigned integer values will be
            hexed. Defaults to False.
            values that are more appropriate for the data type. Defaults to

        Returns:
            pandas.DataFrame: The DataFrame containing the data.
        """
        return pandas_utils.to_dataframe(self, hex_mode, no_nan)

    def _check_struct(self, index: int, data: 'BaseStruct'):
        # check if nan values are present
        for key, value in dataclasses.asdict(data).items():
            if np.isnan(value).any():
                raise ValueError(
                    f"NaN values are not allowed. ({key}"
                    f"of {self.section} #{index:X})"
                )

    def _check_section(self) -> None:
        pass

    def _init_from_dataframe(self, df: pd.DataFrame):
        pandas_utils.from_dataframe(df, self)

    def _init_from_descriptor(self: Self, descriptor: DataDescriptor):
        assert self.__rname__ == descriptor.section, (
            "Cannot init from descriptor of different section."
        )
        setattr(self, "section", descriptor.section)
        setattr(self, "entries", descriptor.entries)

        if descriptor.special_name is not None:
            if type(descriptor.special_name) is list:
                setattr(
                    self, descriptor.special_name[0], descriptor.additional)
                setattr(
                    self, descriptor.special_name[1], descriptor.padding)
            else:
                setattr(self, descriptor.special_name, descriptor.additional)
                setattr(self, "padding", descriptor.padding)
        else:
            setattr(self, "ignored", descriptor.additional)

        setattr(self, "_rdata", descriptor.descriptor)
        _ = [self._check_struct(i, rd) for i, rd in enumerate(self._rdata)]

    def _init_from_parser(self: Self, parser: Parser) -> None:
        assert parser.is_read_contiuously, "Parser is not read continuously"

        section = parser.read_string(4)
        entries = parser.read_uint16()

        # set attributes header info
        setattr(self, "section", section)
        setattr(self, "entries", entries)
        special_name = _SpecialName.get(section, None)
        if type(special_name) is str:
            setattr(self, special_name, parser.read_uint16())
        elif type(special_name) is list and len(special_name) == 2:
            values = parser.read_uint8(2)
            for name, value in zip(special_name, values):
                setattr(self, name, value)
        else:
            setattr(self, "ignored", parser.read_uint16())

        _rdata = _ListMax255()
        struct_cls = self.__struct__

        pass_ret_keys = []
        for k, v in self.__custom_fn__.items():
            v = list(v.ret_keys)
            v.remove(k)
            pass_ret_keys.extend(v)
        for idx in range(entries):
            # read normally
            _init_kwargs = {}
            for key, value in self.__annotations__.items():
                if key in pass_ret_keys:
                    continue
                # check if the key is a custom function
                spec = self.__custom_fn__.get(key, None)
                if spec is not None:
                    spec.set_args(_init_kwargs)
                    for attrk, attv in spec(parser).items():
                        _init_kwargs[attrk] = attv
                    continue
                # n in expected to be an integer
                dt, n = t.get_dtype_and_size(value)
                if not isinstance(n, int):
                    raise TypeError(
                        "Invalid type for n in get_dtype_and_size. "
                        "Use if `custom_fn` instead.`"
                    )
                data = parser.read_number(dt, n)
                _init_kwargs[key] = data
            try:
                struct = struct_cls(**_init_kwargs)
                self._check_struct(idx, struct)
                _rdata.append(struct_cls(**_init_kwargs))
            except TypeError as e:
                if __debug__:
                    raise e
                msg = str(e) + (
                    '\nFor developer: did you match all the names '
                    'in the section and struct class?'
                )
                raise TypeError(msg) from e

        self._rdata = _rdata

    def _to_str(self: Self, brackets: str = '()'):
        lb, rb = brackets
        _str = self.section + lb + 'entries=' + str(self.entries)
        pad_name = _SpecialName.get(self.section, 'ignored') # type: ignore
        if isinstance(pad_name, list):
            _str += (
                ', ' + str(pad_name[0]) + '=' + str(getattr(self, pad_name[0]))
            )
            _str += (
                ', ' + str(pad_name[1]) + '=' + str(getattr(self, pad_name[1]))
            )
        else:
            _str += (', ' + str(pad_name) + '=' + str(getattr(self, pad_name)))

        meta_str = []
        for key, data in self._metadata.items():
            dtype, elements = data
            if isinstance(elements, tuple):
                elements = elements[-1]

            if elements == 1:
                size_str = 'scalar, '
            else:
                size_str = f'{elements} elements, '
            meta_str.append(key + "=(" + size_str + str(dtype) + ")")
        meta_str = ", ".join(meta_str)
        _str += ', ' + meta_str + rb
        return _str

    def _sync_entries(self: Self):
        """Sync `entries` with the length of the data."""
        if hasattr(self, 'total_points'):
            total_points = np.uint16(sum(x.numpoints for x in self._rdata))
            if total_points > 255:
                raise OverflowError(
                    "This section exceeds the maximum number of points "
                    f"({total_points} > 255)."
                )
            self.total_points = total_points
        self.entries = np.uint16(len(self._rdata))

    def __str__(self: Self) -> str:
        if self.__indexing__:
            return self._to_str('()')
        return repr(self._rdata[0])

    __repr__ = __str__

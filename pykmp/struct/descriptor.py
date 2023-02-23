import contextlib
import dataclasses
import warnings
from collections import UserList
from typing import Any, Iterable, Optional

from typing_extensions import Self

import pykmp._typing as t

_SpecialName = {
    'POTI': 'total_points',
    'CAME': ['op_camera', 'padding'],
}


class _ListMax255(UserList):
    def __init__(
        self: Self, data: Optional[Iterable[Any]] = None
    ):
        if data is not None and len(data) > 255:
            raise OverflowError('List length must be less than 256.')
        super().__init__(data)

    @contextlib.contextmanager
    def _check_limit(self: Self, _rstr: str, obj_: Any):
        if (
            (len(self.data) > 255)
            or (
                isinstance(obj_, int) and (
                    (_rstr == 'multiply' and len(self.data) * obj_ > 255)
                    or (_rstr == 'insert' and len(self.data) + obj_ > 255)
                    )
            )
            or (hasattr(obj_, '__len__') and len(self.data) + len(obj_) >= 255)
        ):
            raise OverflowError(f'Cannot {_rstr} to data with length 255.')
        yield
        if len(self.data) == 255:
            warnings.warn(
                'The number of elements has been reached the maximum 255. '
                f'You cannot {_rstr} more elements to this data'
            )

    def _check_type_match(self: Self, item: Any):
        if len(self) == 0:
            return
        obj = [item] if not isinstance(item, Iterable) else item
        for obj_ in obj:
            if not isinstance(obj_, type(self[0])):
                raise TypeError(
                    f'Cannot append {type(obj_)} to a list of {type(self[0])}.'
                )

    def copy(self: Self):
        return self.__class__(list(map(lambda x: x.copy(), self.data)))

    def append(self: Self, item: Any) -> None:
        self._check_type_match(item)
        with self._check_limit('append', item):
            super().append(item)

    def __eq__(self: Self, other: Self):
        return all(o == s for o, s in zip(other, self))

    def __add__(self: Self, other: Iterable[Any]) -> Self:
        self._check_type_match(other)
        with self._check_limit('add', other):
            return super().__add__(other)

    def __radd__(self: Self, other: Iterable[Any]) -> Self:
        self._check_type_match(other)
        with self._check_limit('add', other):
            return super().__radd__(other)

    def __iadd__(self: Self, other: Iterable[Any]) -> Self:
        self._check_type_match(other)
        with self._check_limit('add', other):
            return super().__iadd__(other)

    def __mul__(self: Self, n: int) -> Self:
        with self._check_limit('multiply', n):
            return super().__mul__(n)

    def __imul__(self: Self, n: int) -> Self:
        with self._check_limit('multiply', n):
            return super().__imul__(n)

    def insert(self, i: int, item: Any) -> None:
        self._check_type_match(item)
        with self._check_limit('insert', item):
            return super().insert(i, item)

    def extend(self: Self, other: Iterable[Any]) -> None:
        self._check_type_match(other)
        with self._check_limit('extend', other):
            super().extend(other)


_EQ_IGNORED = (
    'padding',
    'unknown',
    'unknown2',
    'unknown3',
    'unused',
    'unused2',
)


@dataclasses.dataclass(frozen=True, eq=False)
class DataDescriptor:
    section: str
    entries: t.UInt16
    additional: t.UInt16
    padding: Optional[t.UInt16]
    special_name: Optional[str | list[str]]
    descriptor: _ListMax255

    def __eq__(self: Self, other: Self):
        if not isinstance(other, DataDescriptor):
            return False
        return all(
            getattr(self, k) == getattr(other, k)
            for k in dataclasses.asdict(self).keys() if k not in _EQ_IGNORED
        )

    def __post_init__(self: Self):
        if not isinstance(self.descriptor, _ListMax255):
            raise TypeError(
                'descriptor should be a _ListMax255 instance.'
            )
        if self.entries != len(self.descriptor):
            raise ValueError(
                'entries should be the same as the length of descriptor.'
            )

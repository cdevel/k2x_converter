import contextlib
import os
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

_ALLOWED_DTYPES = [np.uint8, np.uint16, np.uint32, np.int16, np.float32]


def _convert_dtype_to_little_endian(dtype: npt.DTypeLike) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype.byteorder == '>':
        return dtype
    dtype = np.dtype(f'>{dtype.kind}{dtype.alignment}')
    return dtype


class _BinaryParser:
    def __init__(self: Self, data: Union[str, bytes]):
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError('File {} does not exist'.format(data))
            with open(data, 'rb') as f:
                self._data = f.read()
            self._size = os.path.getsize(data)
        elif isinstance(data, bytes):
            assert len(data) > 0, 'Data must be non-empty'
            self._data = data
            self._size = len(data)
        self._byte_offset = 0
        self._read_contiuously = False

    def _read(self: Self, buffer_size: int) -> bytes:
        """Private method to read data from the file. do not use this method directly."""
        if buffer_size == 0:
            raise ValueError('You are attempting to read an empty buffer')
        elif buffer_size < 0:
            raise ValueError('Buffer size must be positive')
        start = self.current_offset()
        last = start + buffer_size
        if not __debug__:
            print(f'Reading {buffer_size} bytes from {hex(start)} to {hex(last)}')
        # overflow
        if last > self._size:
            raise OverflowError('You are attempting to read past the end of the file')
        result = self._data[start:last]
        self.seek(last)
        return result

    def _check_buffer_size(self: Self, length: int):
        if not isinstance(length, int):
            raise TypeError('Length must be an integer')
        elif length <= 0:
            raise ValueError('Length must be positive')
        elif length >= self._size:
            raise ValueError('Length of string is too large')

    def _validate_start_and_back(self, start, back):
        if self._read_contiuously:
            return None, False
        return start, back

    def seek(self: Self, offset: int):
        """
        Seek to the given offset.

        Args:
            offset (int): Offset to seek to.

        Raises:
            TypeError: If the offset is not an integer.
            ValueError: If the offset is negative or over the file size.
        """
        if not isinstance(offset, int):
            raise TypeError(f'Offset must be an integer. Got {type(offset)}')
        elif offset < 0:
            raise ValueError(f'Offset must be non-negative. Got {offset}')
        elif offset > self._size:
            raise ValueError(
                'Offset of is too large. '
                f'Got {hex(offset)} (max: {hex(self._size)})'
            )
        self._byte_offset = offset

    def current_offset(self: Self):
        """
        Get the current offset of the file.

        Args:
            as_hex (bool): Whether to return the offset as a hex string. Defaults to False.

        Returns:
            int: the current offset of the file
        """
        return self._byte_offset

    @contextlib.contextmanager
    def read_contiuously(
        self: Self, start: Optional[int] = None, back: bool = True
    ):
        """
        Temporarily set the file to read contiuously.
        """
        if self._read_contiuously:
            raise RuntimeError('File is already set to read contiuously')
        self._read_contiuously = True
        prev_offset = self.current_offset()
        start = start or self.current_offset()
        self.seek(start)
        try:
            yield
        finally:
            self._read_contiuously = False
            if back:
                self.seek(prev_offset)

    @property
    def is_read_contiuously(self: Self) -> bool:
        """Whether the file is set to read contiuously"""
        return self._read_contiuously

    @contextlib.contextmanager
    def seek_scope(self: Self, offset: int, back: bool = False):
        """
        Temporarily seek to the given offset.

        Args:
            offset (int): Offset to seek to.
            back (bool, optional): Whether to seek back to the previous offset after the context is executed. Defaults to False.
        """
        prev_offset = self.current_offset()
        self.seek(offset)
        try:
            yield
        finally:
            if back:
                self.seek(prev_offset)

    def read_string(
        self: Self, size: int, start: Optional[int] = None, back: bool = True
    ) -> str:
        """Read a string from the file

        Args:
            size (int): Length of the data to read.
            start (int, optional): Offset to seek to before reading. Defaults to None.
            back (bool, optional): Whether to seek to the next string after reading. Defaults to False.

        Notes:
            Whitin the scope of `read_contiuously`, arguments `start` and `back` are ignored.

        Raises:
            ValueError: If the string is not valid UTF-8.

        Returns:
            str: the string read from the file
        """
        start, back = self._validate_start_and_back(start, back)
        start = start or self.current_offset()
        with self.seek_scope(start, back):
            result = self._read(size)
            try:
                result = result.decode('utf-8')
            except UnicodeDecodeError:
                raise ValueError(f'String is not valid UTF-8, got {result}')
        return result

    @property
    def header(self: Self) -> str:
        """Read the header of the file, most likely "RKMD".

        Returns:
            str: the header of the file
        """
        return self.read_string(size=4, start=0, back=True)

    def read_number(
        self: Self,
        dtype: npt.DTypeLike,
        n: int = 1,
        start: Optional[int] = None,
        back: bool = True,
        size: Optional[int] = None,
        fillbyte: Optional[bytes] = None,
    ) -> npt.NDArray:
        """
        Read an integer or float from the file.

        Args:
            dtype (str, np.dtype): The type of the value to read.
            n (int, optional): Number of elements to read. Defaults to 1.
            start (int, optional): Offset to seek to before reading. Defaults to None.
            back (bool, optional): Whether to seek to the next string after reading. Defaults to False.
            size (int, optional): Length of the data to read. Defaults to None, which will be inferred from the dtype.
            fillbyte (int or bytes, optional): Bytes to be filled in the read value.
            If specified, argument `size` also needs to be specified. Defaults to None.

        Notes:
            Whitin the scope of `read_contiuously`, arguments `start` and `back` are ignored.

        Raises:
            TypeError: If the dtype is not supported.
            ValueError: Fillbyte is specified but size is not, or fillbyte is too large.

        Returns:
            np.ndarray: the value read from the file, same dtype as the argument `dtype`
        """
        start, back = self._validate_start_and_back(start, back)
        dtype = _convert_dtype_to_little_endian(dtype)
        if dtype.type not in _ALLOWED_DTYPES:
            raise TypeError(
                f'Unsupported dtype. Got {dtype}, '
                f'expected one of {_ALLOWED_DTYPES}'
            )

        # fillbyte check
        if fillbyte is not None and size is None:
            raise ValueError('fillbyte is specified but size is not')

        start = start or self.current_offset()
        with self.seek_scope(start, back):
            if fillbyte is not None and isinstance(size, int):
                # fillbyte is specified
                if len(fillbyte) + size > dtype.alignment:
                    raise ValueError(
                        f'Fillbyte is too large. Got {len(fillbyte)}, '
                        f'expected {dtype.alignment - size}'
                    )
            else:
                fillbyte = b''
                size = size or (dtype.alignment * n)
            result = self._read(size)
            if fillbyte is not None:
                # append fillbyte to the result before converting to numpy array
                result = result + fillbyte
            result = np.frombuffer(result, dtype=dtype, count=n)
        if n == 1:
            return result[0]
        return result

    def read_uint8(
        self: Self,
        n: int = 1,
        start: Optional[int] = None,
        back: bool = True
    ) -> npt.NDArray[np.uint8]:
        """Read an unsigned 8-bit integer from the file.

        Args:
            n (int, optional): Number of elements to read. Defaults to 1.
            start (int, optional): Offset to seek to before reading. Defaults to None.
            back (bool, optional): Whether to seek to the next string after reading. Defaults to False.

        Returns:
            np.ndarray: the value read from the file.
            if `n` is 1, return a scalar, otherwise return an array.
        """
        return self.read_number(
            np.uint8, n=n, start=start, back=back
        )

    def read_uint16(
        self: Self,
        n: int = 1,
        start: Optional[int] = None,
        back: bool = True
    ) -> npt.NDArray[np.uint16]:
        """Read an unsigned 16-bit integer from the file.

        Args:
            n (int, optional): Number of elements to read. Defaults to 1.
            start (int, optional): Offset to seek to before reading. Defaults to None.
            back (bool, optional): Whether to seek to the next string after reading. Defaults to False.

        Returns:
            np.ndarray: the value read from the file.
            if `n` is 1, return a scalar, otherwise return an array.
        """
        return self.read_number(
            np.uint16, n=n, start=start, back=back
        )

    def read_uint32(
        self: Self,
        n: int = 1,
        start: Optional[int] = None,
        back: bool = True
    ) -> npt.NDArray[np.uint32]:
        """Read an unsigned 32-bit integer from the file.

        Args:
            start (int, optional): Offset to seek to before reading. Defaults to None.
            back (bool, optional): Whether to seek to the next string after reading. Defaults to False.
            n (int, optional): Number of elements to read. Defaults to 1.

        Returns:
            np.ndarray: the value read from the file.
            if `n` is 1, return a scalar, otherwise return an array.
        """
        return self.read_number(
            np.uint32, n=n, start=start, back=back
        )

    def read_uint64(
        self: Self,
        n: int = 1,
        start: Optional[int] = None,
        back: bool = True
    ) -> npt.NDArray[np.uint64]:
        """Read an unsigned 64-bit integer from the file.

        Args:
            n (int, optional): Number of elements to read. Defaults to 1.
            start (int, optional): Offset to seek to before reading. Defaults to None.
            back (bool, optional): Whether to seek to the next string after reading. Defaults to False.

        Returns:
            np.ndarray: the value read from the file.
            if `n` is 1, return a scalar, otherwise return an array.
        """
        return self.read_number(
            np.uint64, n=n, start=start, back=back
        )

    def read_int16(
        self: Self,
        n: int = 1,
        start: Optional[int] = None,
        back: bool = True
    ) -> npt.NDArray[np.int16]:
        """Read a signed 16-bit integer from the file.

        Args:
            n (int, optional): Number of elements to read. Defaults to 1.
            start (int, optional): Offset to seek to before reading. Defaults to None.
            back (bool, optional): Whether to seek to the next string after reading. Defaults to False.
            n (int, optional): Number of elements to read. Defaults to 1.

        Returns:
            np.ndarray: the value read from the file.
            if `n` is 1, return a scalar, otherwise return an array.
        """
        return self.read_number(
            np.int16, n=n, start=start, back=back
        )

    def read_float32(
        self: Self,
        n: int = 1,
        start: Optional[int] = None,
        back: bool = True,
        fillbyte: Optional[bytes] = None
    ) -> npt.NDArray[np.float32]:
        """Read a 32-bit float from the file.

        Args:
            n (int, optional): Number of elements to read. Defaults to 1.
            start (int, optional): Offset to seek to before reading. Defaults to None.
            back (bool, optional): Whether to seek to the next string after reading. Defaults to False.
            fillbyte (bytes, optional): Fillbyte to append to the end of the result. Defaults to None.

        Returns:
            np.ndarray: the value read from the file.
            if `n` is 1, return a scalar, otherwise return an array.
        """
        return self.read_number(
            np.float32, n=n, start=start, back=back, fillbyte=fillbyte
        )

    # alias
    read_short = read_int16
    read_float = read_float32

# alias
Parser = _BinaryParser

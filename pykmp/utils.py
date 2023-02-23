from struct import pack as fpack

import numpy as np
import numpy.typing as npt


def correct_order(v):
    """Fix the byteorder if dtype is float."""
    # TODO: add if v is not a numpy array
    if v.ndim > 0 and v.dtype.kind == 'f':
        val = v.astype('>f4')
    else:
        val = v
    return val


def tobytes(
    value: np.ndarray | int | float | str | list | tuple,
    dtype_hint: npt.DTypeLike | None = None,
) -> bytes:
    """
    Convert a value to bytes.

    Args:
        value: The value to convert.
        dtype_hint (np.dtype, optional): The dtype of the value.
        Must be specified if the value is int.

    Returns:
        bytes: The value converted to bytes.
    """

    if dtype_hint is not None:
        dtype_hint = np.dtype(dtype_hint)
        if isinstance(value, (int, float)):
            value = float(value) if dtype_hint.kind == "f" else int(value)

    if isinstance(value, int):
        if dtype_hint is None:
            raise ValueError("Cannot convert int to bytes without dtype_hint.")
        try:
            ret = int(value).to_bytes(
                dtype_hint.itemsize, "big", signed=dtype_hint.kind == "i")
        except OverflowError as e:
            iinfo = np.iinfo(dtype_hint)
            raise ValueError(
                "int value is out of range for dtype. "
                f"Expected [{iinfo.min}, {iinfo.max}], "
                f"got {value}."
            ) from e
        return ret
    elif (
        isinstance(value, float)
        or (isinstance(value, np.floating) and value.ndim == 0)
    ):
        return fpack(">f", float(value))
    elif isinstance(value, np.integer) and value.ndim == 0:
        return int(value).to_bytes(
            value.dtype.itemsize, "big", signed=value.dtype.kind == "i")
    elif isinstance(value, str):
        return value.encode("utf-8")
    elif isinstance(value, (list, tuple)):
        value = np.array(value, dtype=dtype_hint)

    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"Unsupported type: {type(value)}."
        )

    # np.ndarray
    if value.dtype.kind == 'f':
        value = value.astype('>f4')
    else:
        value = value.byteswap()
    return value.tobytes()

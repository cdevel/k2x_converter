import dataclasses
from typing import Any, Callable, Sequence

from typing_extensions import Self

from pykmp._io._parser import _BinaryParser as Parser
from pykmp._typing import DataClass


def inplace_or_copy(cls, copy: bool, func):
    if copy:
        return func(cls.copy())
    return func(cls)


@dataclasses.dataclass
class CustomFnSpec:
    """
    Special function specification for custom parsing.
    Private use only. You don't need to use this class.

    Args:
        fn (callable): A function to parse. first argument must be a parser.
        ret_keys (list): A list of keys to return.
        additional_args (list): A list of additional argument "name" for `fn`.
    """
    fn: Callable[[Parser, Any], dict[str, Any]]
    ret_keys: Sequence[str]
    additional_args: Sequence[str] = dataclasses.field(default_factory=tuple)

    def __post_init__(self: Self):
        self._args = {}

    def set_args(self: Self, keydict: dict[str, Any]):
        """
        Set additional arguments for the function.
        If `additional_args` is empty, this method does nothing.

        Args:
            keydict (dict): A dict of key and value.
            args should be in the dict.
        """
        if not self.additional_args or not keydict:
            return
        self._args = {k: keydict.get(k) for k in self.additional_args}

    def __call__(self: Self, parser: Parser) -> dict[str, Any]:
        if self._args:
            ret = self.fn(parser, **self._args)
            self._args.clear()
        else:
            ret = self.fn(parser)
        return ret


def section_add_attrs(
    struct: DataClass,
    indexing: bool = True,
    custom_fn: dict[str, CustomFnSpec] | None = None,
):
    """
    Add attributes to a `BaseSection` class.
    Private use only. You don't need to use this decorator.

    Args:
        struct (Class): A class that has `__annotations__` attribute.
        offset (int): Section offset of kmp. See https://wiki.tockdom.com/wiki/KMP_(File_Format)#Mario_Kart_Wii_specific_file_header
        indexing (bool): Whether the struct can use indexing.
        Only STR0 or WIM0 is assumed.
        custom_fn (dict, optional): A dict of CustomFnSpec.
        Only GOBJ, POTI or STGI is assumed.
    """
    def wrapper(cls):
        if not dataclasses.is_dataclass(struct):
            raise TypeError('struct should be a dataclass.')
        cls.__struct__ = struct
        cls.__indexing__ = indexing
        cls.__rname__ = cls.__name__

        if not (custom_fn is None or isinstance(custom_fn, dict)):
            raise TypeError(
                "custom_fn should be a dict of CustomFnSpec. or None.")
        elif (custom_fn is not None
            and not all(
                    isinstance(k, CustomFnSpec) for k in custom_fn.values()
                )
            ):
                raise TypeError(
                    "custom_fn should be a dict of CustomFnSpec."
                )
        if custom_fn:
            # check duplicate ret_keys
            ret_keys = set()
            for spec in custom_fn.values():
                for key in spec.ret_keys:
                    if key in ret_keys:
                        raise ValueError(
                            f"Duplicate ret_key {key} in custom_fn."
                        )
                    ret_keys.add(key)
        cls.__custom_fn__ = custom_fn or {}
        return cls
    return wrapper


# def _parse_path(parser: Parser):
#     """
#     start, length -> start
#     """
#     assert parser.is_read_contiuously
#     start = parser.read_uint8()
#     _ = parser.read_uint8() # length
#     return {'start': start}


# PATH_SPEC = {
#     'start': CustomFnSpec(_parse_path, ('start',)),
# }

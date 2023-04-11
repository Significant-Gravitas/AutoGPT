from functools import partial
import inspect

from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    overload,
    Union,
    Tuple,
    Type,
    TypeVar,
)


T = TypeVar("T")


Result = Iterable[Union[Any, Tuple[Any], Tuple[str, Any], Tuple[str, Any, Any]]]
RichReprResult = Result


class ReprError(Exception):
    """An error occurred when attempting to build a repr."""


@overload
def auto(cls: Optional[T]) -> T:
    ...


@overload
def auto(*, angular: bool = False) -> Callable[[T], T]:
    ...


def auto(
    cls: Optional[T] = None, *, angular: Optional[bool] = None
) -> Union[T, Callable[[T], T]]:
    """Class decorator to create __repr__ from __rich_repr__"""

    def do_replace(cls: Type[T], angular: Optional[bool] = None) -> Type[T]:
        def auto_repr(self: Type[T]) -> str:
            """Create repr string from __rich_repr__"""
            repr_str: List[str] = []
            append = repr_str.append

            angular = getattr(self.__rich_repr__, "angular", False)  # type: ignore
            for arg in self.__rich_repr__():  # type: ignore
                if isinstance(arg, tuple):
                    if len(arg) == 1:
                        append(repr(arg[0]))
                    else:
                        key, value, *default = arg
                        if key is None:
                            append(repr(value))
                        else:
                            if len(default) and default[0] == value:
                                continue
                            append(f"{key}={value!r}")
                else:
                    append(repr(arg))
            if angular:
                return f"<{self.__class__.__name__} {' '.join(repr_str)}>"
            else:
                return f"{self.__class__.__name__}({', '.join(repr_str)})"

        def auto_rich_repr(self: Type[T]) -> Result:
            """Auto generate __rich_rep__ from signature of __init__"""
            try:
                signature = inspect.signature(self.__init__)  ## type: ignore
                for name, param in signature.parameters.items():
                    if param.kind == param.POSITIONAL_ONLY:
                        yield getattr(self, name)
                    elif param.kind in (
                        param.POSITIONAL_OR_KEYWORD,
                        param.KEYWORD_ONLY,
                    ):
                        if param.default == param.empty:
                            yield getattr(self, param.name)
                        else:
                            yield param.name, getattr(self, param.name), param.default
            except Exception as error:
                raise ReprError(
                    f"Failed to auto generate __rich_repr__; {error}"
                ) from None

        if not hasattr(cls, "__rich_repr__"):
            auto_rich_repr.__doc__ = "Build a rich repr"
            cls.__rich_repr__ = auto_rich_repr  # type: ignore

        auto_repr.__doc__ = "Return repr(self)"
        cls.__repr__ = auto_repr  # type: ignore
        if angular is not None:
            cls.__rich_repr__.angular = angular  # type: ignore
        return cls

    if cls is None:
        return partial(do_replace, angular=angular)  # type: ignore
    else:
        return do_replace(cls, angular=angular)  # type: ignore


@overload
def rich_repr(cls: Optional[T]) -> T:
    ...


@overload
def rich_repr(*, angular: bool = False) -> Callable[[T], T]:
    ...


def rich_repr(
    cls: Optional[T] = None, *, angular: bool = False
) -> Union[T, Callable[[T], T]]:
    if cls is None:
        return auto(angular=angular)
    else:
        return auto(cls)


if __name__ == "__main__":

    @auto
    class Foo:
        def __rich_repr__(self) -> Result:
            yield "foo"
            yield "bar", {"shopping": ["eggs", "ham", "pineapple"]}
            yield "buy", "hand sanitizer"

    foo = Foo()
    from pip._vendor.rich.console import Console

    console = Console()

    console.rule("Standard repr")
    console.print(foo)

    console.print(foo, width=60)
    console.print(foo, width=30)

    console.rule("Angular repr")
    Foo.__rich_repr__.angular = True  # type: ignore

    console.print(foo)

    console.print(foo, width=60)
    console.print(foo, width=30)

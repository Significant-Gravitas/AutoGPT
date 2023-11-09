import inspect
import types
import typing as t
from functools import update_wrapper
from gettext import gettext as _

from .core import Argument
from .core import Command
from .core import Context
from .core import Group
from .core import Option
from .core import Parameter
from .globals import get_current_context
from .utils import echo

F = t.TypeVar("F", bound=t.Callable[..., t.Any])
FC = t.TypeVar("FC", bound=t.Union[t.Callable[..., t.Any], Command])


def pass_context(f: F) -> F:
    """Marks a callback as wanting to receive the current context
    object as first argument.
    """

    def new_func(*args, **kwargs):  # type: ignore
        return f(get_current_context(), *args, **kwargs)

    return update_wrapper(t.cast(F, new_func), f)


def pass_obj(f: F) -> F:
    """Similar to :func:`pass_context`, but only pass the object on the
    context onwards (:attr:`Context.obj`).  This is useful if that object
    represents the state of a nested system.
    """

    def new_func(*args, **kwargs):  # type: ignore
        return f(get_current_context().obj, *args, **kwargs)

    return update_wrapper(t.cast(F, new_func), f)


def make_pass_decorator(
    object_type: t.Type, ensure: bool = False
) -> "t.Callable[[F], F]":
    """Given an object type this creates a decorator that will work
    similar to :func:`pass_obj` but instead of passing the object of the
    current context, it will find the innermost context of type
    :func:`object_type`.

    This generates a decorator that works roughly like this::

        from functools import update_wrapper

        def decorator(f):
            @pass_context
            def new_func(ctx, *args, **kwargs):
                obj = ctx.find_object(object_type)
                return ctx.invoke(f, obj, *args, **kwargs)
            return update_wrapper(new_func, f)
        return decorator

    :param object_type: the type of the object to pass.
    :param ensure: if set to `True`, a new object will be created and
                   remembered on the context if it's not there yet.
    """

    def decorator(f: F) -> F:
        def new_func(*args, **kwargs):  # type: ignore
            ctx = get_current_context()

            if ensure:
                obj = ctx.ensure_object(object_type)
            else:
                obj = ctx.find_object(object_type)

            if obj is None:
                raise RuntimeError(
                    "Managed to invoke callback without a context"
                    f" object of type {object_type.__name__!r}"
                    " existing."
                )

            return ctx.invoke(f, obj, *args, **kwargs)

        return update_wrapper(t.cast(F, new_func), f)

    return decorator


def pass_meta_key(
    key: str, *, doc_description: t.Optional[str] = None
) -> "t.Callable[[F], F]":
    """Create a decorator that passes a key from
    :attr:`click.Context.meta` as the first argument to the decorated
    function.

    :param key: Key in ``Context.meta`` to pass.
    :param doc_description: Description of the object being passed,
        inserted into the decorator's docstring. Defaults to "the 'key'
        key from Context.meta".

    .. versionadded:: 8.0
    """

    def decorator(f: F) -> F:
        def new_func(*args, **kwargs):  # type: ignore
            ctx = get_current_context()
            obj = ctx.meta[key]
            return ctx.invoke(f, obj, *args, **kwargs)

        return update_wrapper(t.cast(F, new_func), f)

    if doc_description is None:
        doc_description = f"the {key!r} key from :attr:`click.Context.meta`"

    decorator.__doc__ = (
        f"Decorator that passes {doc_description} as the first argument"
        " to the decorated function."
    )
    return decorator


CmdType = t.TypeVar("CmdType", bound=Command)


@t.overload
def command(
    __func: t.Callable[..., t.Any],
) -> Command:
    ...


@t.overload
def command(
    name: t.Optional[str] = None,
    **attrs: t.Any,
) -> t.Callable[..., Command]:
    ...


@t.overload
def command(
    name: t.Optional[str] = None,
    cls: t.Type[CmdType] = ...,
    **attrs: t.Any,
) -> t.Callable[..., CmdType]:
    ...


def command(
    name: t.Union[str, t.Callable[..., t.Any], None] = None,
    cls: t.Optional[t.Type[Command]] = None,
    **attrs: t.Any,
) -> t.Union[Command, t.Callable[..., Command]]:
    r"""Creates a new :class:`Command` and uses the decorated function as
    callback.  This will also automatically attach all decorated
    :func:`option`\s and :func:`argument`\s as parameters to the command.

    The name of the command defaults to the name of the function with
    underscores replaced by dashes.  If you want to change that, you can
    pass the intended name as the first argument.

    All keyword arguments are forwarded to the underlying command class.
    For the ``params`` argument, any decorated params are appended to
    the end of the list.

    Once decorated the function turns into a :class:`Command` instance
    that can be invoked as a command line utility or be attached to a
    command :class:`Group`.

    :param name: the name of the command.  This defaults to the function
                 name with underscores replaced by dashes.
    :param cls: the command class to instantiate.  This defaults to
                :class:`Command`.

    .. versionchanged:: 8.1
        This decorator can be applied without parentheses.

    .. versionchanged:: 8.1
        The ``params`` argument can be used. Decorated params are
        appended to the end of the list.
    """

    func: t.Optional[t.Callable[..., t.Any]] = None

    if callable(name):
        func = name
        name = None
        assert cls is None, "Use 'command(cls=cls)(callable)' to specify a class."
        assert not attrs, "Use 'command(**kwargs)(callable)' to provide arguments."

    if cls is None:
        cls = Command

    def decorator(f: t.Callable[..., t.Any]) -> Command:
        if isinstance(f, Command):
            raise TypeError("Attempted to convert a callback into a command twice.")

        attr_params = attrs.pop("params", None)
        params = attr_params if attr_params is not None else []

        try:
            decorator_params = f.__click_params__  # type: ignore
        except AttributeError:
            pass
        else:
            del f.__click_params__  # type: ignore
            params.extend(reversed(decorator_params))

        if attrs.get("help") is None:
            attrs["help"] = f.__doc__

        cmd = cls(  # type: ignore[misc]
            name=name or f.__name__.lower().replace("_", "-"),  # type: ignore[arg-type]
            callback=f,
            params=params,
            **attrs,
        )
        cmd.__doc__ = f.__doc__
        return cmd

    if func is not None:
        return decorator(func)

    return decorator


@t.overload
def group(
    __func: t.Callable[..., t.Any],
) -> Group:
    ...


@t.overload
def group(
    name: t.Optional[str] = None,
    **attrs: t.Any,
) -> t.Callable[[F], Group]:
    ...


def group(
    name: t.Union[str, t.Callable[..., t.Any], None] = None, **attrs: t.Any
) -> t.Union[Group, t.Callable[[F], Group]]:
    """Creates a new :class:`Group` with a function as callback.  This
    works otherwise the same as :func:`command` just that the `cls`
    parameter is set to :class:`Group`.

    .. versionchanged:: 8.1
        This decorator can be applied without parentheses.
    """
    if attrs.get("cls") is None:
        attrs["cls"] = Group

    if callable(name):
        grp: t.Callable[[F], Group] = t.cast(Group, command(**attrs))
        return grp(name)

    return t.cast(Group, command(name, **attrs))


def _param_memo(f: FC, param: Parameter) -> None:
    if isinstance(f, Command):
        f.params.append(param)
    else:
        if not hasattr(f, "__click_params__"):
            f.__click_params__ = []  # type: ignore

        f.__click_params__.append(param)  # type: ignore


def argument(*param_decls: str, **attrs: t.Any) -> t.Callable[[FC], FC]:
    """Attaches an argument to the command.  All positional arguments are
    passed as parameter declarations to :class:`Argument`; all keyword
    arguments are forwarded unchanged (except ``cls``).
    This is equivalent to creating an :class:`Argument` instance manually
    and attaching it to the :attr:`Command.params` list.

    :param cls: the argument class to instantiate.  This defaults to
                :class:`Argument`.
    """

    def decorator(f: FC) -> FC:
        ArgumentClass = attrs.pop("cls", None) or Argument
        _param_memo(f, ArgumentClass(param_decls, **attrs))
        return f

    return decorator


def option(*param_decls: str, **attrs: t.Any) -> t.Callable[[FC], FC]:
    """Attaches an option to the command.  All positional arguments are
    passed as parameter declarations to :class:`Option`; all keyword
    arguments are forwarded unchanged (except ``cls``).
    This is equivalent to creating an :class:`Option` instance manually
    and attaching it to the :attr:`Command.params` list.

    :param cls: the option class to instantiate.  This defaults to
                :class:`Option`.
    """

    def decorator(f: FC) -> FC:
        # Issue 926, copy attrs, so pre-defined options can re-use the same cls=
        option_attrs = attrs.copy()
        OptionClass = option_attrs.pop("cls", None) or Option
        _param_memo(f, OptionClass(param_decls, **option_attrs))
        return f

    return decorator


def confirmation_option(*param_decls: str, **kwargs: t.Any) -> t.Callable[[FC], FC]:
    """Add a ``--yes`` option which shows a prompt before continuing if
    not passed. If the prompt is declined, the program will exit.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--yes"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """

    def callback(ctx: Context, param: Parameter, value: bool) -> None:
        if not value:
            ctx.abort()

    if not param_decls:
        param_decls = ("--yes",)

    kwargs.setdefault("is_flag", True)
    kwargs.setdefault("callback", callback)
    kwargs.setdefault("expose_value", False)
    kwargs.setdefault("prompt", "Do you want to continue?")
    kwargs.setdefault("help", "Confirm the action without prompting.")
    return option(*param_decls, **kwargs)


def password_option(*param_decls: str, **kwargs: t.Any) -> t.Callable[[FC], FC]:
    """Add a ``--password`` option which prompts for a password, hiding
    input and asking to enter the value again for confirmation.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--password"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """
    if not param_decls:
        param_decls = ("--password",)

    kwargs.setdefault("prompt", True)
    kwargs.setdefault("confirmation_prompt", True)
    kwargs.setdefault("hide_input", True)
    return option(*param_decls, **kwargs)


def version_option(
    version: t.Optional[str] = None,
    *param_decls: str,
    package_name: t.Optional[str] = None,
    prog_name: t.Optional[str] = None,
    message: t.Optional[str] = None,
    **kwargs: t.Any,
) -> t.Callable[[FC], FC]:
    """Add a ``--version`` option which immediately prints the version
    number and exits the program.

    If ``version`` is not provided, Click will try to detect it using
    :func:`importlib.metadata.version` to get the version for the
    ``package_name``. On Python < 3.8, the ``importlib_metadata``
    backport must be installed.

    If ``package_name`` is not provided, Click will try to detect it by
    inspecting the stack frames. This will be used to detect the
    version, so it must match the name of the installed package.

    :param version: The version number to show. If not provided, Click
        will try to detect it.
    :param param_decls: One or more option names. Defaults to the single
        value ``"--version"``.
    :param package_name: The package name to detect the version from. If
        not provided, Click will try to detect it.
    :param prog_name: The name of the CLI to show in the message. If not
        provided, it will be detected from the command.
    :param message: The message to show. The values ``%(prog)s``,
        ``%(package)s``, and ``%(version)s`` are available. Defaults to
        ``"%(prog)s, version %(version)s"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    :raise RuntimeError: ``version`` could not be detected.

    .. versionchanged:: 8.0
        Add the ``package_name`` parameter, and the ``%(package)s``
        value for messages.

    .. versionchanged:: 8.0
        Use :mod:`importlib.metadata` instead of ``pkg_resources``. The
        version is detected based on the package name, not the entry
        point name. The Python package name must match the installed
        package name, or be passed with ``package_name=``.
    """
    if message is None:
        message = _("%(prog)s, version %(version)s")

    if version is None and package_name is None:
        frame = inspect.currentframe()
        f_back = frame.f_back if frame is not None else None
        f_globals = f_back.f_globals if f_back is not None else None
        # break reference cycle
        # https://docs.python.org/3/library/inspect.html#the-interpreter-stack
        del frame

        if f_globals is not None:
            package_name = f_globals.get("__name__")

            if package_name == "__main__":
                package_name = f_globals.get("__package__")

            if package_name:
                package_name = package_name.partition(".")[0]

    def callback(ctx: Context, param: Parameter, value: bool) -> None:
        if not value or ctx.resilient_parsing:
            return

        nonlocal prog_name
        nonlocal version

        if prog_name is None:
            prog_name = ctx.find_root().info_name

        if version is None and package_name is not None:
            metadata: t.Optional[types.ModuleType]

            try:
                from importlib import metadata  # type: ignore
            except ImportError:
                # Python < 3.8
                import importlib_metadata as metadata  # type: ignore

            try:
                version = metadata.version(package_name)  # type: ignore
            except metadata.PackageNotFoundError:  # type: ignore
                raise RuntimeError(
                    f"{package_name!r} is not installed. Try passing"
                    " 'package_name' instead."
                ) from None

        if version is None:
            raise RuntimeError(
                f"Could not determine the version for {package_name!r} automatically."
            )

        echo(
            t.cast(str, message)
            % {"prog": prog_name, "package": package_name, "version": version},
            color=ctx.color,
        )
        ctx.exit()

    if not param_decls:
        param_decls = ("--version",)

    kwargs.setdefault("is_flag", True)
    kwargs.setdefault("expose_value", False)
    kwargs.setdefault("is_eager", True)
    kwargs.setdefault("help", _("Show the version and exit."))
    kwargs["callback"] = callback
    return option(*param_decls, **kwargs)


def help_option(*param_decls: str, **kwargs: t.Any) -> t.Callable[[FC], FC]:
    """Add a ``--help`` option which immediately prints the help page
    and exits the program.

    This is usually unnecessary, as the ``--help`` option is added to
    each command automatically unless ``add_help_option=False`` is
    passed.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--help"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """

    def callback(ctx: Context, param: Parameter, value: bool) -> None:
        if not value or ctx.resilient_parsing:
            return

        echo(ctx.get_help(), color=ctx.color)
        ctx.exit()

    if not param_decls:
        param_decls = ("--help",)

    kwargs.setdefault("is_flag", True)
    kwargs.setdefault("expose_value", False)
    kwargs.setdefault("is_eager", True)
    kwargs.setdefault("help", _("Show this message and exit."))
    kwargs["callback"] = callback
    return option(*param_decls, **kwargs)

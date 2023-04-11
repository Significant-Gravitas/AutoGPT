from __future__ import annotations

import importlib
import pkgutil
import threading
from typing import Any, Callable, Optional

import tiktoken_ext

from tiktoken.core import Encoding

_lock = threading.RLock()
ENCODINGS: dict[str, Encoding] = {}
ENCODING_CONSTRUCTORS: Optional[dict[str, Callable[[], dict[str, Any]]]] = None


def _find_constructors() -> None:
    global ENCODING_CONSTRUCTORS
    with _lock:
        if ENCODING_CONSTRUCTORS is not None:
            return
        ENCODING_CONSTRUCTORS = {}

        # tiktoken_ext is a namespace package
        # submodules inside tiktoken_ext will be inspected for ENCODING_CONSTRUCTORS attributes
        # - we use namespace package pattern so `pkgutil.iter_modules` is fast
        # - it's a separate top-level package because namespace subpackages of non-namespace
        #   packages don't quite do what you want with editable installs
        plugin_mods = pkgutil.iter_modules(tiktoken_ext.__path__, tiktoken_ext.__name__ + ".")

        for _, mod_name, _ in plugin_mods:
            mod = importlib.import_module(mod_name)
            try:
                constructors = mod.ENCODING_CONSTRUCTORS
            except AttributeError as e:
                raise ValueError(
                    f"tiktoken plugin {mod_name} does not define ENCODING_CONSTRUCTORS"
                ) from e
            for enc_name, constructor in constructors.items():
                if enc_name in ENCODING_CONSTRUCTORS:
                    raise ValueError(
                        f"Duplicate encoding name {enc_name} in tiktoken plugin {mod_name}"
                    )
                ENCODING_CONSTRUCTORS[enc_name] = constructor


def get_encoding(encoding_name: str) -> Encoding:
    if encoding_name in ENCODINGS:
        return ENCODINGS[encoding_name]

    with _lock:
        if encoding_name in ENCODINGS:
            return ENCODINGS[encoding_name]

        if ENCODING_CONSTRUCTORS is None:
            _find_constructors()
            assert ENCODING_CONSTRUCTORS is not None

        if encoding_name not in ENCODING_CONSTRUCTORS:
            raise ValueError(f"Unknown encoding {encoding_name}")

        constructor = ENCODING_CONSTRUCTORS[encoding_name]
        enc = Encoding(**constructor())
        ENCODINGS[encoding_name] = enc
        return enc


def list_encoding_names() -> list[str]:
    with _lock:
        if ENCODING_CONSTRUCTORS is None:
            _find_constructors()
            assert ENCODING_CONSTRUCTORS is not None
        return list(ENCODING_CONSTRUCTORS)

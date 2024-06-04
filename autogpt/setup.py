from shutil import which
from pkgutil import iter_modules

from cx_Freeze import setup, Executable

import os

packages = [
    m.name for m in iter_modules() if m.ispkg and ("poetry" in m.module_finder.path)
]

print(packages)


setup(
    executables=[
        Executable("autogpt/__main__.py", target_name="autogpt"),
        # Executable(which("hypercorn"), target_name="hypercorn"),
    ],
    options={
        "build_exe": {
            "packages": packages,
            "includes": [
                "autogpt",
                "spacy",
                "spacy.lang",
                "spacy.vocab",
                "spacy.lang.lex_attrs",
                "uvicorn.loops.auto",
                "srsly.msgpack.util",
                "blis",
                "uvicorn.protocols.http.auto",
                "uvicorn.protocols.websockets.auto",
                "uvicorn.lifespan.on",
            ],
            "excludes": ["readability.compat.two"],
        },
        "bdist_mac": {
            "bundle_name": "AutoGPT",
            "include_resources": [""],
        },
    },
)

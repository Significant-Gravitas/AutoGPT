from shutil import which
from pkgutil import iter_modules

# from cx_Freeze import setup, Executable

# import os

# packages = [
#     m.name for m in iter_modules() if m.ispkg and ".venv" in m.module_finder.path
# ]

# setup(
#     executables=[
#         Executable(("autogpt_server/app.py"), target_name="server", base="service")
#     ],
#     options={
#         "build_exe": {
#             "packages": packages,
#             "includes": [
#                 "autogpt_server",
#             ],
#         },
#         "bdist_mac": {"bundle_name": "AutoGPT", "include_resources": ["IMG_3775.jpeg"]},
#     },
# )

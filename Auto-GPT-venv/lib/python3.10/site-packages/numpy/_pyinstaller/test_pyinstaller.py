import subprocess
from pathlib import Path

import pytest


# PyInstaller has been very unproactive about replacing 'imp' with 'importlib'.
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
# It also leaks io.BytesIO()s.
@pytest.mark.filterwarnings('ignore::ResourceWarning')
@pytest.mark.parametrize("mode", ["--onedir", "--onefile"])
@pytest.mark.slow
def test_pyinstaller(mode, tmp_path):
    """Compile and run pyinstaller-smoke.py using PyInstaller."""

    pyinstaller_cli = pytest.importorskip("PyInstaller.__main__").run

    source = Path(__file__).with_name("pyinstaller-smoke.py").resolve()
    args = [
        # Place all generated files in ``tmp_path``.
        '--workpath', str(tmp_path / "build"),
        '--distpath', str(tmp_path / "dist"),
        '--specpath', str(tmp_path),
        mode,
        str(source),
    ]
    pyinstaller_cli(args)

    if mode == "--onefile":
        exe = tmp_path / "dist" / source.stem
    else:
        exe = tmp_path / "dist" / source.stem / source.stem

    p = subprocess.run([str(exe)], check=True, stdout=subprocess.PIPE)
    assert p.stdout.strip() == b"I made it!"

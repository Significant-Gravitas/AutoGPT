import subprocess
from pathlib import Path

artifacts = Path("e2e/artifacts")
device_file = artifacts / "simulator-id.txt"
if device_file.exists():
    log_path = artifacts / "simulator-errors.log"
    with log_path.open("wb") as output:
        try:
            subprocess.run(
                [
                    "xcrun",
                    "simctl",
                    "spawn",
                    device_file.read_text().strip(),
                    "log",
                    "show",
                    "--style",
                    "compact",
                    "--last",
                    "1m",
                    "--predicate",
                    "(messageType == error OR messageType == fault) AND "
                    '(process == "SpringBoard" OR process == "ReportCrash" OR '
                    'process == "diagnosticd" OR process CONTAINS "Poster" OR '
                    'process == "apsd")',
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=10,
                check=True,
            )
        except (subprocess.SubprocessError, OSError) as error:
            print(f"Simulator error collection incomplete: {error}")
    with log_path.open("rb") as output:
        output.seek(max(0, log_path.stat().st_size - 512 * 1024))
        tail = output.read()
    log_path.write_bytes(tail)

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def extract_verified(item, archive_path, destination):
    digest = "sha256:" + hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert digest == item["digest"], item["name"]
    prefix, shard = item["name"].split("-py", 1)[1].split("-", 1)
    assert prefix in {"3.12", "3.13"}
    expected = {
        f".coverage-{shard}",
        f"coverage-{shard}.xml",
        f"junit-{shard}.xml",
        f"shard-metrics-{shard}.json",
    }
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == expected
        assert len(archive.namelist()) == len(expected)
        assert archive.testzip() is None
        for name in expected:
            assert Path(name).name == name
            (destination / name).write_bytes(archive.read(name))


def download(item, reports):
    destination = reports / item["name"]
    destination.mkdir()
    archive = destination / "download.zip"
    with archive.open("wb") as stream:
        subprocess.run(
            [
                "gh",
                "api",
                "repos/Significant-Gravitas/AutoGPT/actions/artifacts/"
                f"{item['id']}/zip",
            ],
            stdout=stream,
            check=True,
            timeout=90,
        )
    extract_verified(item, archive, destination)
    archive.unlink()
    return item["name"]


if __name__ == "__main__":
    assert os.environ.get("GITHUB_ACTIONS") == "true" and sys.platform == "linux"
    manifest, target = map(Path, sys.argv[1:])
    items = json.loads(manifest.read_text())
    assert len(items) == len({item["id"] for item in items}) == 66
    target.mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        completed = list(pool.map(lambda item: download(item, target), items))
    assert len(completed) == len(set(completed)) == 66
    print("Downloaded and SHA-256 verified all 66 exact artifacts.")

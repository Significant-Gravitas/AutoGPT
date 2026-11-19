import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_backend_tools import compare_inventories

root = Path(sys.argv[1])
expected = {
    f"platform-single-container-ci-{arch}-{installer}"
    for arch in ("amd64", "arm64")
    for installer in ("poetry", "uv")
}
assert {path.name for path in root.iterdir()} == expected
trees = set()
parents = set()
inventories = {}
for arch in ("amd64", "arm64"):
    for installer in ("poetry", "uv"):
        report = root / f"platform-single-container-ci-{arch}-{installer}"
        status = json.loads((report / "status.json").read_text())
        metadata = json.loads((report / "installer.json").read_text())
        assert status["validated"] is True
        assert status["sha"] == metadata["sha"] == os.environ["GITHUB_SHA"]
        assert metadata["arch"] == arch and metadata["installer"] == installer
        assert status["platform"] == f"linux/{arch}"
        parents.add(metadata["parent"])
        trees.add(metadata["application_tree"])
        inventories[arch, installer] = json.loads(
            (report / "inventory.json").read_text()
        )
        assert len(inventories[arch, installer]) == metadata["inventory_count"] > 100
    differences = compare_inventories(
        inventories[arch, "poetry"], inventories[arch, "uv"]
    )
    print(json.dumps({"arch": arch, "differences": differences}, indent=2))
    assert not differences
assert len(parents) == len(trees) == 1
print("All four complete appliance validations passed with locked-package parity.")

import json
import subprocess
import time
from tempfile import NamedTemporaryFile

STILL_PROCESSING = "File is still processing. Check back later."


def test_file_cli() -> None:
    contents = json.dumps({"prompt": "1 + 3 =", "completion": "4"}) + "\n"
    with NamedTemporaryFile(suffix=".jsonl", mode="wb") as train_file:
        train_file.write(contents.encode("utf-8"))
        train_file.flush()
        create_output = subprocess.check_output(
            ["openai", "api", "files.create", "-f", train_file.name, "-p", "fine-tune"]
        )
    file_obj = json.loads(create_output)
    assert file_obj["bytes"] == len(contents)
    file_id: str = file_obj["id"]
    assert file_id.startswith("file-")
    start_time = time.time()
    while True:
        delete_result = subprocess.run(
            ["openai", "api", "files.delete", "-i", file_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        if delete_result.returncode == 0:
            break
        elif STILL_PROCESSING in delete_result.stderr:
            time.sleep(0.5)
            if start_time + 60 < time.time():
                raise RuntimeError("timed out waiting for file to become available")
            continue
        else:
            raise RuntimeError(
                f"delete failed: stdout={delete_result.stdout} stderr={delete_result.stderr}"
            )

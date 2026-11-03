import json
import os
import re
import subprocess

for label, job in [("appliance", "126049402760"), ("fullstack", "126050596724"), ("backend", "126050601660")]:
    result = subprocess.run(["gh", "api", f"repos/Significant-Gravitas/AutoGPT/actions/jobs/{job}/logs"], capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    revisions = []
    for index, line in enumerate(lines):
        if "git log -1 --format=%H" in line:
            revisions.extend(re.findall(r"\b[0-9a-f]{40}\b", "\n".join(lines[index+1:index+3])))
        if "args.VCS_REF=" in line:
            revisions.extend(re.findall(r"args.VCS_REF=([0-9a-f]{40})", line))
    print("::notice title=source-revision::" + json.dumps({"job": job, "label": label, "revisions": revisions}))

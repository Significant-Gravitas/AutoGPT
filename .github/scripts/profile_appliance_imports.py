import json
import os
import re
import subprocess
import time
from pathlib import Path

output = Path(os.environ['PROFILE_OUTPUT'])
output.mkdir(parents=True)
modules = ['backend.rest', 'backend.db', 'backend.scheduler', 'backend.notification',
           'backend.batch_executor', 'backend.exec', 'backend.copilot.executor.__main__',
           'backend.ws', 'backend.cli.publish_skills_catalog']
results = []
code = "import importlib,sys,time,json; started=time.monotonic(); importlib.import_module(sys.argv[1]); print(json.dumps({'module':sys.argv[1],'import_seconds':time.monotonic()-started}))"
for module in modules:
    started = time.monotonic()
    container = 'ci-import-profile-' + module.replace('.', '-')
    command = ['docker', 'run', '--rm', '--name', container, os.environ['PROFILE_IMAGE'], 'bash', '-c',
               'exec runuser --user autogpt -- python -X importtime -c "$1" "$2"',
               'profile', code, module]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as error:
        (output / f'{module}.error.txt').write_text(str(error))
        raise
    finally:
        subprocess.run(['docker', 'rm', '-f', '-v', container], capture_output=True)
    (output / f'{module}.stdout.txt').write_text(result.stdout)
    (output / f'{module}.stderr.txt').write_text(result.stderr)
    entries = []
    for line in result.stderr.splitlines():
        match = re.match(r'import time:\s*(\d+)\s*\|\s*(\d+)\s*\| (.*)', line)
        if match:
            entries.append({'self_us': int(match[1]), 'cumulative_us': int(match[2]),
                            'module': match[3].strip()})
    row = {'module': module, 'exit_code': result.returncode,
           'container_and_import_seconds': round(time.monotonic() - started, 3),
           'slowest_self': sorted(entries, key=lambda x: x['self_us'], reverse=True)[:15],
           'slowest_cumulative': sorted(entries, key=lambda x: x['cumulative_us'], reverse=True)[:15]}
    results.append(row)
    print('::notice title=appliance-import::' + json.dumps(row))
    (output / 'results.json').write_text(json.dumps(results, indent=2))
assert all(row['exit_code'] == 0 for row in results), results

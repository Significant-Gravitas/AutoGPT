import concurrent.futures
import json
import os
import pathlib
import re
import subprocess

output = pathlib.Path(os.environ['RUNNER_TEMP']) / 'ci-measurements'
output.mkdir()


def gh(*args):
    return subprocess.run(['gh', *args], check=True, text=True, capture_output=True).stdout


def annotation(title, data):
    value = json.dumps(data, separators=(',', ':'))
    value = value.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
    print(f'::notice title={title}::{value}')


def architecture():
    folder = output / 'architecture'
    gh('run', 'download', '41675168027', '--pattern', 'architecture-comparison-*', '--dir', str(folder))
    result = {}
    for path in folder.rglob('comparison.json'):
        data = json.loads(path.read_text())
        record = {}
        for label, values in data.items():
            text = (path.parent / f'resources-{label}.txt').read_text()
            match = re.search(r'Maximum resident set size \(kbytes\): (\d+)', text)
            record[label] = {'seconds': values['seconds'], 'cases': len(values['cases']), 'rss_kb': int(match[1]) if match else None, 'metrics': json.loads((path.parent / f'metrics-{label}.json').read_text())}
        result[path.parent.name] = record
    return 'architecture-prefilter', result


def imports():
    folder = output / 'imports'
    gh('run', 'download', '41673605349', '--name', 'python-imports-3.12', '--dir', str(folder))
    result = []
    for path in folder.rglob('python-imports.log'):
        for line in path.read_text(errors='replace').splitlines():
            match = re.match(r'import time:\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.*)', line)
            if match:
                result.append({'self_us': int(match[1]), 'cumulative_us': int(match[2]), 'module': match[3].strip()})
    return 'python312-imports', sorted(result, key=lambda item: item['cumulative_us'], reverse=True)[:35]


def job_log(label, run, job):
    log = gh('run', 'view', run, '--job', job, '--log')
    (output / f'{label}.log').write_text(log)
    rows = []
    for line in log.splitlines():
        text = line.split('\t')[-1]
        timestamp = text.split(' ', 1)[0]
        match = re.search(r'#(\d+) (.+)', text)
        if match and (re.search(r'(DONE \d+|CACHED|exporting|importing cache|load build context|RUN |COPY )', match[2])):
            rows.append(timestamp + ' #' + match[1] + ' ' + match[2][:220])
        elif re.search(r'Cache (restored|not found|hit)|single-container (smoke timing|boot milestone)|Successfully restored cache|combine_seconds|launcher_to_configure', text):
            rows.append(text[:500])
    return label, rows


tasks = [architecture, imports,
         lambda: job_log('appliance-amd64', '41673952463', '126049402760'),
         lambda: job_log('smoke-shard2', '41673952967', '126050596724'),
         lambda: job_log('backend-util312', '41673952967', '126050601660')]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for future in concurrent.futures.as_completed([pool.submit(task) for task in tasks]):
        title, data = future.result()
        (output / f'{title}.json').write_text(json.dumps(data, indent=2))
        if isinstance(data, list) and len(data) > 70:
            for start in range(0, len(data), 70):
                annotation(f'{title}-{start // 70 + 1}', data[start:start+70])
        else:
            annotation(title, data)

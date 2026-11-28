import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / "cla-label-sync.yml"
STEP_NAME = "Sync CLA labels and handle stale PRs"

WARNING = "<!-- cla-close-warning -->"
CLOSED = "<!-- cla-auto-closed -->"

# Runs the step's script the way actions/github-script does, against a fake
# GitHub holding one platform PR, and prints what it wrote.
HARNESS = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const sc = input.scenario;
const DAY = 24 * 60 * 60 * 1000;
const now = Date.now();
const ago = days => new Date(now - days * DAY).toISOString();
const calls = { comments: [], updates: [], errors: [], failed: [] };
const claStates = [...sc.cla_states];
const pr = () => ({
  number: 1, state: 'open', user: { type: 'User' }, created_at: ago(sc.pr_age_days),
  head: { sha: 'abc' }, labels: sc.labels.map(name => ({ name })),
});
const github = {
  paginate: async (fn, params) => (await fn(params)).data,
  rest: {
    pulls: {
      list: async () => ({ data: [pr()] }),
      get: async () => ({ data: pr() }),
      listFiles: async () => ({ data: [{ filename: 'autogpt_platform/backend/x.py' }] }),
      update: async p => { calls.updates.push(p); return { data: {} }; },
    },
    repos: {
      getCombinedStatusForRef: async () => ({ data: { statuses: [{
        context: 'license/cla', description: '',
        state: claStates.length > 1 ? claStates.shift() : claStates[0],
      }] } }),
    },
    checks: { listForRef: async () => ({ data: { check_runs: [] } }) },
    issues: {
      listComments: async () => ({ data: sc.comments.map(([body, days]) => ({
        user: { type: 'Bot' }, body, created_at: ago(days),
      })) }),
      listEvents: async () => ({ data: sc.signed_label_days.map(days => ({
        event: 'labeled', label: { name: 'cla: signed' }, created_at: ago(days),
      })) }),
      createComment: async p => { calls.comments.push(p.body); return { data: {} }; },
      addLabels: async () => ({ data: {} }),
      removeLabel: async () => ({ data: {} }),
    },
  },
};
const context = { eventName: 'schedule', repo: { owner: 'o', repo: 'r' }, payload: {} };
const core = { setFailed: m => calls.failed.push(m), warning: () => {}, info: () => {} };
console.log = () => {};
console.error = (...args) => calls.errors.push(args.join(' '));
Object.assign(process.env, input.env);
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
new AsyncFunction('github', 'context', 'core', 'require', input.script)(github, context, core, require)
  .then(() => process.stdout.write(JSON.stringify(calls)));
"""

# name, scenario, expected (close warning posted, PR closed)
CASES = [
    (
        "a PR past day 30 with no warning is warned, not closed",
        dict(labels=["cla: signed"], pr_age_days=40, comments=[]),
        (True, False),
    ),
    (
        "a warning younger than the grace period holds the close",
        dict(labels=["cla: pending"], pr_age_days=40, comments=[(WARNING, 3)]),
        (False, False),
    ),
    (
        "a warning past the grace period closes a PR still unsigned",
        dict(labels=["cla: pending"], pr_age_days=40, comments=[(WARNING, 8)]),
        (False, True),
    ),
    (
        "a fresh read that comes back signed holds the close",
        dict(
            labels=["cla: pending"],
            pr_age_days=40,
            comments=[(WARNING, 8)],
            cla_states=["pending", "success"],
        ),
        (False, False),
    ),
    (
        "a warning from before the author signed is renewed, not acted on",
        dict(
            labels=["cla: signed"],
            pr_age_days=90,
            comments=[(WARNING, 60)],
            signed_label_days=[50],
        ),
        (True, False),
    ),
    (
        "a PR auto-closed before and reopened is not warned again",
        dict(labels=["cla: pending"], pr_age_days=40, comments=[(CLOSED, 5)]),
        (False, False),
    ),
]


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class CloseDecisionTests(unittest.TestCase):
    def test_close_decisions(self):
        script, env = step_script_and_env()
        env["DEMOTE_DEBOUNCE_SECONDS"] = "0"
        for name, scenario, expected in CASES:
            with self.subTest(name):
                calls = run_script(script, env, scenario)
                self.assertEqual((calls["errors"], calls["failed"]), ([], []))
                warned = any(WARNING in body for body in calls["comments"])
                closed = any(u.get("state") == "closed" for u in calls["updates"])
                self.assertEqual((warned, closed), expected)


def run_script(script: str, env: dict, scenario: dict) -> dict:
    scenario = {"cla_states": ["pending"], "signed_label_days": [], **scenario}
    result = subprocess.run(
        ["node", "-e", HARNESS],
        input=json.dumps({"script": script, "env": env, "scenario": scenario}),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


def step_script_and_env() -> tuple:
    lines = WORKFLOW.read_text().splitlines()
    env_start = lines.index("env:") + 1
    env = {}
    for line in lines[env_start:]:
        if line and not line.startswith(" "):
            break
        match = re.match(r"  ([A-Z_]+):\s*(?:'([^']*)'|(\S+))", line)
        if match:
            env[match.group(1)] = match.group(2) or match.group(3)

    step = next(i for i, line in enumerate(lines) if f"- name: {STEP_NAME}" in line)
    start = next(i for i in range(step, len(lines)) if lines[i].strip() == "script: |")
    body = lines[start + 1 :]
    indent = next(len(line) - len(line.lstrip()) for line in body if line.strip())
    script = []
    for line in body:
        if line.strip() and len(line) - len(line.lstrip()) < indent:
            break
        script.append(line[indent:])
    return "\n".join(script), env


if __name__ == "__main__":
    unittest.main()

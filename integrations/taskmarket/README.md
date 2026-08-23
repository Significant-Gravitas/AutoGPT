# Taskmarket Requester Integration

Lets an AutoGPT user complete the Taskmarket *requester* workflow from inside the agent:
- configure (Base network enforced, 8453)
- create a task with explicit description / reward / deadline / deliverables
- require FRESH user authorization before any funded on-chain action
- retrieve live task status + submissions for human review (never auto-accept)

Uses the official `taskmarket` CLI as first-party tooling. No private keys or secrets handled here.

## Safeguards
- Network restricted to Base (8453)
- `authorized_by_user` required before funding
- `reward <= max_spend` enforced
- settlement status never blindly retried

## Usage
```python
import sys; sys.path.insert(0, 'integrations/taskmarket')
import requester as tm
tm.configure()  # Base only
tm.create_task("My task", 2.0, deadline_unix, "deliverable",
               max_spend_usdc=5.0, authorized_by_user=True)
status = tm.get_status(task_id)
subs = tm.get_submissions(task_id)  # for human review
```

Requires `npm install -g @lucid-agents/taskmarket`.

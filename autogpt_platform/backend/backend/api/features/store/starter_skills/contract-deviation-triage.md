---
name: "contract-deviation-triage"
description: "Sort contract deviations by the user's playbook priority and route each one to its business owner or counsel."
triggers: ["triage contract deviations", "contract exceptions", "redline issues", "playbook deviations", "legal review queue"]
version: "1"
---

# Contract deviation triage

Use a completed playbook comparison. Triage is queue order, not legal risk advice.

## Apply only supplied rules

If the user's playbook names escalation classes, thresholds, owners, or fallbacks, apply them and cite the rule. If it does not, use neutral process labels:

- `COUNSEL DECISION` — a legal position or unsupplied fallback is needed;
- `BUSINESS INPUT` — price, scope, service, timing, or owner fact is missing;
- `PLAYBOOK MATCH` — the supplied position matches;
- `DOCUMENT GAP` — text, definition, exhibit, or version is missing;
- `READY FOR CONFIRMATION` — all requested evidence is present, but no approval is implied.

Never create high, medium, or low legal-risk grades unless the supplied playbook defines them. Never infer authority from a job title.

## Output

Return one row per issue with source section, playbook rule, label, owner, deadline, blocker, supplied fallback, and exact decision. Keep all approvals pending until the named owner records them. Do not accept text, negotiate, or send changes.

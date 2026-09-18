---
name: "nda-playbook-review"
description: "Compare an NDA clause by clause with the user's supplied NDA playbook and prepare deviations for counsel."
triggers: ["review nda", "nda redline", "confidentiality agreement", "nda playbook", "mutual nda"]
version: "1"
---

# NDA playbook review

Use the complete NDA and the user's supplied NDA playbook. Without the playbook, extract clauses and questions only.

## Compare

Cover parties, purpose, confidential-information definition, exclusions, use and disclosure, representatives, compelled disclosure, security, return or destruction, retained copies, term, confidentiality duration, residuals, intellectual property, remedies, publicity, assignment, governing law, venue, and signature blocks when present.

For each item show:

- agreement section and exact text;
- supplied playbook position and exact text;
- `MATCH`, `DEVIATION`, `MISSING`, or `UNCLEAR`;
- neutral description of the difference;
- supplied fallback, if one exists;
- business fact or counsel decision still needed.

Do not infer that silence is acceptance. Do not call a clause mutual, standard, reasonable, enforceable, or risky unless that is a label in the supplied playbook, and even then attribute it to the playbook.

## Output and gate

Return the comparison table and counsel queue. Any proposed redline must use a supplied fallback and remain an unsent draft. Counsel chooses the position and approves the text.

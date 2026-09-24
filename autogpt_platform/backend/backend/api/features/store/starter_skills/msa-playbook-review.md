---
name: "msa-playbook-review"
description: "Compare an MSA and linked order terms with the user's supplied playbook, citing every deviation for counsel."
triggers: ["review msa", "msa redline", "services agreement review", "msa playbook", "master agreement"]
version: "1"
---

# MSA playbook review

Use the complete MSA, order form, statements of work, data terms, security terms, amendments, and the user's supplied playbook. State which documents are missing.

## Compare by topic

Review only topics covered by the supplied playbook, such as scope and order priority, fees, taxes, payment, service levels, acceptance, changes, intellectual property, licence, data, security, confidentiality, warranties, indemnities, liability, insurance, audit, compliance, term, suspension, termination, transition, publicity, assignment, notices, governing law, and dispute process.

For each topic cite the document, section, agreement text, playbook text, label, difference, supplied fallback, and decision owner. Track conflicts between the MSA and linked documents instead of choosing which controls.

## Limits

Do not calculate legal exposure, judge enforceability, label market practice, or recommend acceptance. A proposed edit must come from an exact supplied fallback. Anything else becomes a question for counsel.

## Output

Return the deviation table, document conflicts, missing schedules, business inputs, and counsel decisions needed before signature. Do not send a redline or accept a term.

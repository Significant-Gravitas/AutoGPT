---
name: "contract-key-term-extraction"
description: "Extract contract facts into a source-cited tracker without interpreting ambiguous language or giving legal advice."
triggers: ["extract contract terms", "contract summary", "key terms tracker", "contract metadata", "agreement terms"]
version: "1"
---

# Contract key-term extraction

Extract what the document says; do not decide what it means when the wording is unclear.

## Record

Capture document name and version, legal parties, effective date, signature date, start, end, initial term, renewal, notice period and method, fees, currency, payment timing, price changes, scope, service levels, credits, data locations, insurance, intellectual-property ownership, confidentiality period, liability text, indemnity topics, suspension, termination, transition, assignment, governing law, venue, and order of precedence when present.

Every value carries the document, section, exact source passage, and extraction status: `STATED`, `CALCULATED`, `MISSING`, `CONFLICT`, or `UNCLEAR`. For a calculated notice date, show the source date, period, and formula. Keep conflicts across amendments visible.

## Limits

Do not turn missing text into "none". Do not resolve a conflict, choose a controlling document, or state a legal consequence. Send unclear dates, defined terms, and document priority to counsel or the contract owner.

## Output

Return the tracker, missing documents, conflicts, and review queue. The summary is an index to the source, not a replacement for it.

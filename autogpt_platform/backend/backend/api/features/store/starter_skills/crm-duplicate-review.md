---
name: "crm-duplicate-review"
description: "Find likely duplicate records with stated match rules and evidence, for a person to merge."
triggers: ["duplicates", "duplicate contacts", "merge records", "dedupe CRM", "duplicate accounts"]
version: "1"
---

# CRM duplicate review

Use this for contacts, leads, or accounts from a dated export.

## State the match rules

Write the rules before matching, for example exact email, same domain and
surname, or same company name after removing "Ltd" and "Inc". Give each rule a
strength: **strong**, **likely**, or **weak**. Record the export date and row
count the rules ran against.

## Show the pairs

For each pair or group show:

- record ids and owners;
- the fields that match and the fields that differ;
- the rule that matched and its strength;
- open deals, activity, and last touch on each record;
- which record looks like the survivor, and why.

Keep groups with open deals or different owners apart; they need a person to
look first.

## Merge list for approval

Return three lists: strong matches ready for review, likely matches that need a
check, and pairs you ruled out. Give counts for each. Say which reports will
change once the merges are done.

Never merge, delete, or edit a record yourself, and do not guess which value
is correct when two records disagree. A person approves and makes each merge.

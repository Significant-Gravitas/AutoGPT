---
name: "investor-targeting-and-research"
description: "Build a sourced investor target list from published stage, sector, geography, cheque range, portfolio, and conflict facts."
triggers: ["find investors", "investor list", "fundraising research", "VC research", "target investors"]
version: "1"
---

# Investor targeting and research

Use public or user-supplied facts to prepare research. This is outreach planning,
not investment advice and not proof that an investor will invest.

## Set the target

Record the company's stage, sector, geography, amount sought, expected cheque
range, lead or follow role, timing, and any excluded conflicts. Use only terms
the founder or approved materials provide.

## Research each candidate

Return one row per firm or investor:

- exact name and source URL;
- published stage, sector, geography, and cheque range with source date;
- relevant investments and dates;
- stated lead or follow preference, if published;
- possible conflict with the reason and source;
- named partner only when a current public source links them to the thesis;
- fit score: strong, possible, or weak;
- fit reason and missing fact;
- last checked date.

Separate firm facts from partner facts. A prior investment does not prove a
current thesis, available capital, or interest. Mark stale pages and conflicting
sources.

## Rank without false certainty

Use the approved target criteria and show the score components. Do not rank on
prestige alone. Put a candidate in `needs review` when a conflict, cheque range,
or stage fit remains unclear. Keep referrals and relationship notes as
user-supplied facts, not public facts.

## Safety rules

Never invent a person, email, investment, fund status, cheque size, or interest.
Do not scrape private data, infer protected traits, recommend an investment, or
promise access or a meeting. Do not send outreach. Flag confidential company
details before using them in any draft, and require owner approval for the
audience and content.

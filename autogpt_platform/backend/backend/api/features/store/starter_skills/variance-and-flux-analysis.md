---
name: "variance-and-flux-analysis"
description: "Use when the user asks what moved, why actuals missed plan, or what changed period over period — the driver decomposition, the red/yellow/green grades, and the open items carried forward."
triggers: ["what moved", "why did we miss plan", "flux commentary", "variance read", "period over period change", "explain the gap to budget", "actual vs plan"]
version: "1"
---

# Variance and flux analysis

Run this when the user wants to know what moved and why. Every claim
carries a number; no number means UNKNOWN, not a guess.

## Inputs

Two comparable columns from the finance ledger: actual against plan,
actual against forecast, the current forecast against the prior one, or
this period against last.

## Fix the period and the comparison first

State both in one line before anything else. Never mix a partial month
against a full one without saying so — a half-month against a full plan
is not a variance, it is a timing artifact.

## Decompose the move into drivers

Price against volume against mix against timing; headcount against
rate; one-time against run rate. Three to six lines, each with its own
number. No claim without a number, and no filler attribution — "higher
spend" is not a driver.

## Sort the lines into blocks

Over-plan, under-plan, and flat-or-timing, with the gap in currency AND
in percent on every line, each labeled favorable (F) or unfavorable
(UF). For expenses, favorable means budget minus actual is positive.
Skip lines under the quiet floor — the default stops a line firing
under 500 in the user's currency.

## Grade each line and name its owner

Red, yellow, or green against the variance threshold, with the owner
named. Green is inside plan or inside the threshold; yellow is past the
threshold but recoverable this period; red needs an owner decision. The
default threshold — the greater of 10% or 5,000 in the user's currency
— suits a small team; the enterprise norm is above 50,000 or above 5%.
Yellow gets a watch line. Red gets the one question the owner has to
answer.

## Call out thin samples

A line with one or two data points gets flagged as too early to call,
not explained anyway. A confident story on two data points is the fast
route to a wrong forecast.

## Write it down and carry it forward

Write the read dated under the variance reads, carry open items forward
until an owner closes them, and feed repeat variances back into the
next forecast refresh so the same miss stops recurring. Reads taken
before the hard close are labeled soft-close preliminary.

## Output

The comparison line, the driver blocks, the red/yellow/green grades
with owners, and the open-items carry list.

## Fallbacks

The ledger lacks a comparable period: say which side is UNKNOWN and
read only the side you have. A driver needs data outside the ledger:
name the source and ask once instead of guessing.

## Approval gate

Reads and drafts only. Commentary attached to a close, a board pack, or
anything leaving the team waits for the owner's yes.

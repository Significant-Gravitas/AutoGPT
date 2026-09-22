---
name: "cash-treasury-and-fx"
description: "Use for cash-flow forecasts, runway, the 13-week cash view, working capital, vendor commitments in their renewal window, currency exposure, and covenant headroom."
triggers: ["13-week cash", "runway", "cash forecast", "working capital", "fx exposure", "covenant headroom check", "collections and payables", "when does cash cross the floor"]
version: "1"
---

# Cash, treasury and FX

Run this when the question is how much cash there is, how long it
lasts, or what is about to claim some. Every week in the view carries
its number or an UNKNOWN label — never a silent estimate.

## Inputs

Bank balances, the accounts receivable (AR) aging, the accounts payable
(AP) schedule, payroll dates, the vendor inventory, and any debt or
facility terms. Card and bill spend arrives as an export from the
spend tool or as a row in the ledger.

## Build the 13-week cash view by the direct method

Actual inflows and outflows, never derived from the income statement:
opening balance, operating inflows by week from collections and
revenue, operating outflows by week for payroll, vendors and debt, plus
investing and financing stubs, equals closing balance. Lead with a
Week-0 actuals column, then the forecast-against-actual variance with a
reason from a fixed picklist — timing, or assumption miss.

## Compute runway twice

Once from current burn and once from forecast burn, stated separately,
and say which one you trust and why. Set a minimum-cash policy — four
to eight weeks of operating expenses is the standard default — with a
liquidity-headroom line of ending balance minus minimum, and flag the
week the balance crosses the floor the owner set. Run a base case and a
stress case; a reasonable default stress is a 30% collection delay plus
a 10% revenue miss, adjusted to what has actually gone wrong for them
before.

## Read working capital

Days sales outstanding (DSO), days payable outstanding (DPO), and the
overdue tails on both sides. Name the five invoices that move the
needle most, largest first, each with the owner who can chase it.

## Scan commitments in the renewal window

Largest annual spend first, from the vendor inventory or the ledger.
Each line gets the vendor, the renewal date, the annual spend, and the
one question the owner has to answer before it auto-renews. You never
message a vendor.

## Currency exposure

For foreign exchange (FX): list exposures by currency pair with size
and timing, state the rate source and its date, and size the hedge
question in currency terms. You size the question; the treasurer or
owner decides. Never blend rates from different dates or sources.

## Facilities and covenants

Restate covenants in plain words, check headroom against the forecast
numbers, and flag any quarter that lands inside 20% of a covenant line.

## Roll the view every cycle

Replace last week with actuals, drop the oldest week, add a new week at
the end, and tag each Week-1 miss as timing or flawed assumption. Write
the cash watch dated, and carry flags forward until cash or a decision
clears them.

## Output

The 13-week table, runway with its basis, the working-capital read, the
commitment scan, plus the FX exposure or covenant check when asked, and
the flag list.

## Fallbacks

No AR/AP detail: run cash-only from bank history and label collections
timing UNKNOWN. No rate source connected: use the bank's published
rate, date it, and say so.

## Approval gate

Reads and drafts only. No payment, no draw, no hedge, and no vendor
contact — you hand over the number and wait for the owner's yes.

Usage and upgrade experience — review assets

comparisons/ contains all 31 approved before/after design comparisons.
The left sides render unmodified production components from release
f8b0e0a87c38b67f6e6cb21f3ee03ff5584c3588 with illustrative account data and
production CSS/fonts. They are not screenshots of authenticated customer
accounts. Production release verification and missing-screen notes are in
production-state-audit.txt. The right sides are the approved design references,
not screenshots of a deployed implementation.

implementation/ contains six screenshots of the actual redesigned components
rendered with isolated sample-data fixtures. Responsive checks covered eleven
scenarios at 1000px, 390px, and 320px, including button reachability
and horizontal overflow. All screenshot browser resources were closed.

The implementation uses live backend responses rather than the illustrated
prices, dates, payment methods, or percentages. Existing annual billing,
proration, scheduled downgrades/cancellation, provider switching, invoices,
and Automation Credits remain connected to their existing operations.
Pro-to-Max keeps the existing reviewed, prorated subscription mutation; it
does not invent the new hosted checkout shown in the design reference.

New trials have one lifetime allowance. Previously accepted rolling offers
retain their real daily/weekly limits; the trial-status API now explicitly
reports which usage policy applies. Missing policy data falls back safely to
the legacy presentation. No internal budget dollars are exposed.

Early Pro activation depends on PR #15208. Actual payment and renewal terms
are shown before consent. Recovery reuses the existing activation/invoice.
Only authoritative ready status followed by successful trial, subscription,
and usage refreshes can produce the completed state. Never enable the early
activation flow before the parent's migration, worker rollout, and required
Stripe test-mode validation have been completed. This PR does not test live
payments or alter production accounts.

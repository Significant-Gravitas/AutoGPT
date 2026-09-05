# PR #14361 screenshot assets

Documentation-only branch for https://github.com/Significant-Gravitas/AutoGPT/pull/14361. These images are not included in the implementation branch or its code diff.

- before-fix.png: user-supplied screenshot from the deployed PR preview showing the transaction-history failure.
- after-table.jpg and after-receipt.jpg: browser captures of the actual PR components at implementation commit 9806a83d1a01e518ddf7a13743d79e00237b4ad2, rendered locally with synthetic API fixtures. These are not live account data and do not establish deployed-backend verification.

The runtime fix uses the validated Prisma execution status directly. All 51 focused backend tests pass, including enrichment tests using real Prisma execution models; all seven status values were checked at the JSON response boundary.

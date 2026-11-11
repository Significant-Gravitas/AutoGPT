"""Erasing a closed account and everything it holds (SECRT-2854).

`routine.erase_account` is the entry point: the `erase-account` admin CLI and
the periodic job (`sweep.py`) both call it, for the accounts closed from
Settings, the paywall or by email (`backend/data/account_deletion.py`). See
`docs/engineering/account-deletion.md` for what it deletes and keeps.
"""

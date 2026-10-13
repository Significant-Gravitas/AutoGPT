"""Deleting an account and everything it holds (SECRT-2854).

`routine.delete_account` is the entry point: the `delete-account` admin CLI and
the self-service endpoint (SECRT-2877) both call it. See
`docs/engineering/account-deletion.md` for what it deletes and keeps.
"""

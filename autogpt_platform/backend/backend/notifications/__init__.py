"""Email notifications.

Deliberately empty of re-exports: `backend.data.execution` imports the run
scorer from here, so an eager import of the service would close an import
cycle. Import the concrete module you need.
"""

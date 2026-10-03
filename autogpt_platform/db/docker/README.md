# Legacy Supabase Docker stack

AutoGPT no longer runs Supabase. Self-hosted installs use the plain Postgres `db` service in `autogpt_platform/docker-compose.yml` or the single-container image (`significantgravitas/autogpt`), so you don't need to start anything here.

This compose file is kept for the backend test stack (`autogpt_platform/backend/docker-compose.test.yaml`). If you are upgrading an install that predates the switch, your old database is still in `volumes/db/data`; see "Upgrading an existing (Supabase-based) installation" in the [self-hosting guide](https://docs.agpt.co/platform/self-hosting/getting-started).

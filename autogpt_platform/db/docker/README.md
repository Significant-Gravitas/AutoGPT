# Local database data

Nothing in the platform reads this folder. Its `.gitignore` keeps database and
storage data under `volumes/` out of Git. If you have data there, follow the
[upgrade instructions](../../../docs/platform/getting-started.md#upgrading-an-existing-supabase-based-installation)
before moving or deleting it.

The platform's database is the `db` service in
[`docker-compose.yml`](../../docker-compose.yml), set up by
[`db/init/00-init.sql`](../init/00-init.sql).

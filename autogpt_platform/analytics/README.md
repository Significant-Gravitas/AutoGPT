# AutoGPT Analytics

Pre-built SQL views that expose safe, documented analytics data from the production Postgres database.
All views live in the `analytics` schema and are readable by the `analytics_readonly` role —
nothing outside that schema is exposed.

## Why this exists

- **Security**: raw tables contain sensitive user content (agent inputs/outputs, emails).
  Views expose only the columns needed for analytics, and complex retention queries
  pre-aggregate so no raw user content ever leaves the view.
- **Reusability**: Looker, PostHog Data Warehouse, Otto (via Supabase MCP), and any
  future BI tool all connect to the same `analytics.*` views instead of duplicating
  complex queries.
- **Maintainability**: each view lives as a documented `.sql` file in `queries/`.
  Updating a query is a one-line change + one command to redeploy.

## Directory layout

```
analytics/
  setup.sql           # one-time schema + role + grant setup
  generate_views.py   # reads queries/*.sql → CREATE OR REPLACE VIEW
  views.sql           # pre-generated output of generate_views.py (for reference)
  queries/            # one .sql file per view, with full documentation headers
    auth_activities.sql
    users_activities.sql
    graph_execution.sql
    node_block_execution.sql
    user_block_spending.sql
    user_onboarding.sql
    user_onboarding_funnel.sql
    user_onboarding_integration.sql
    retention_login_weekly.sql
    retention_login_daily.sql
    retention_login_onboarded_weekly.sql
    retention_execution_weekly.sql
    retention_execution_daily.sql
    retention_agent.sql
```

## Available views

| View | Description | Source |
|------|-------------|--------|
| `auth_activities` | Auth events (login/logout/SSO) from Supabase audit log | `auth.audit_log_entries` |
| `users_activities` | Per-user signup, last-seen, execution counts | `auth.sessions`, `platform.AgentGraphExecution` |
| `graph_execution` | Every agent graph execution with status and timing | `platform.AgentGraphExecution` |
| `node_block_execution` | Per-block execution stats within a run | `platform.AgentNodeExecution` |
| `user_block_spending` | Block-level credit spend per user | `platform.AgentNodeExecution` |
| `user_onboarding` | Signup → first execution funnel, per user | multiple platform tables |
| `user_onboarding_funnel` | Aggregated onboarding funnel step counts | `user_onboarding` base |
| `user_onboarding_integration` | Which integrations users connected during onboarding | `platform.UserIntegration` |
| `retention_login_weekly` | Weekly cohort retention based on login events | `auth.sessions` |
| `retention_login_daily` | Daily cohort retention based on login events | `auth.sessions` |
| `retention_login_onboarded_weekly` | Weekly retention restricted to onboarded users | `auth.sessions`, `platform.AgentGraphExecution` |
| `retention_execution_weekly` | Weekly cohort retention based on execution events | `platform.AgentGraphExecution` |
| `retention_execution_daily` | Daily cohort retention based on execution events | `platform.AgentGraphExecution` |
| `retention_agent` | Retention measured by agent-level return usage | `platform.AgentGraphExecution` |

Each `.sql` file in `queries/` has a full documentation header covering:
- Description
- Source tables
- Output columns with types and meaning
- Example queries
- Looker data source alias (for cross-referencing existing Looker reports)

## One-time setup

Run `setup.sql` once in the Supabase SQL Editor as the `postgres` superuser:

```sql
-- Paste the contents of setup.sql into Supabase SQL Editor and run
```

This creates:
1. The `analytics` schema
2. The `analytics_readonly` role with a password you set
3. `GRANT USAGE + SELECT` on `auth`, `platform`, and `analytics` schemas

> **Note**: grants on `auth.sessions` and `auth.audit_log_entries` require the
> `postgres` superuser role. In Supabase this is available in the SQL Editor by default.

## Deploying / updating views

```bash
# Dry-run — prints all SQL without touching the database
python generate_views.py --dry-run

# Apply all views to the database
DATABASE_URL="postgresql://analytics_readonly:YOURPASS@db.<ref>.supabase.co:5432/postgres" \
  python generate_views.py

# Apply only specific views (e.g. after editing one query)
python generate_views.py \
  --only retention_login_weekly,retention_login_daily \
  --db-url "postgresql://..."
```

Requirements: `pip install psycopg2-binary`

The script uses `CREATE OR REPLACE VIEW` so it is safe to re-run at any time.
After running, it also re-issues `GRANT SELECT ON ALL TABLES IN SCHEMA analytics`
so `analytics_readonly` automatically sees new views.

## Connecting tools

### Supabase MCP (recommended for Otto/AI agents)

Use the `analytics_readonly` credentials and restrict the MCP connection to the
`analytics` schema only. The agent can then `SELECT` from any `analytics.*` view
but cannot reach raw platform or auth data.

Connection string:
```
postgresql://analytics_readonly:YOURPASS@db.<ref>.supabase.co:5432/postgres
```

Set the MCP server's schema search path to `analytics` only.

### PostHog Data Warehouse

1. Add a Postgres source in PostHog → Data Warehouse → Sources
2. Use the direct connection host: `db.<ref>.supabase.co:5432`
3. Set the schema to `analytics`
4. Allowlist PostHog EU IPs: `3.75.65.221`, `18.197.246.42`, `3.120.223.253`

PostHog will sync the view data on a schedule. HogQL queries then run against
the synced copies (avoiding the need to support `generate_series`, CTEs, etc.).

### Looker / Looker Studio

Point existing data sources at `analytics.<view_name>` instead of the raw query.
No query changes needed — the view returns the same columns as before.

## Updating a query

1. Edit the relevant file in `queries/`
2. Run `python generate_views.py --only <view_name> --db-url "..."`
3. Commit both the `.sql` change and the regenerated `views.sql`

To regenerate `views.sql` (the combined reference file):

```bash
python generate_views.py --dry-run > views.sql
```

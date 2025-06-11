# Test Data Scripts

This directory contains scripts for creating and updating test data in the AutoGPT Platform database, specifically designed to test the materialized views for the store functionality.

## Scripts

### test_data_creator.py
Creates a comprehensive set of test data including:
- Users with profiles
- Agent graphs, nodes, and executions
- Store listings with multiple versions
- Reviews and ratings
- Library agents
- Integration webhooks
- Onboarding data
- Credit transactions

**Image/Video Domains Used:**
- Images: `picsum.photos` (for all image URLs)
- Videos: `youtube.com` (for store listing videos)

### test_data_updater.py
Updates existing test data to simulate real-world changes:
- Adds new agent graph executions
- Creates new store listing reviews
- Updates store listing versions
- Adds credit transactions
- Refreshes materialized views

## Materialized Views

The scripts test two materialized views:

1. **mv_agent_run_counts**: Tracks execution counts by agent
2. **mv_review_stats**: Tracks review statistics (count, average rating) by store listing

These views are automatically refreshed every 15 minutes via pg_cron, or can be manually refreshed.

## Usage

### Prerequisites

1. Ensure the database is running:
```bash
docker compose up -d
# or for test database:
docker compose -f docker-compose.test.yaml up -d
```

2. Run database migrations:
```bash
poetry run prisma migrate deploy
```

### Running the Scripts

#### Option 1: Use the helper script (from backend directory)
```bash
poetry run python run_test_data.py
```

#### Option 2: Run individually (from backend/test directory)
```bash
# Create initial test data
poetry run python test_data_creator.py

# Update data to test materialized view changes
poetry run python test_data_updater.py
```

#### Option 3: Use the shell script (from backend directory)
```bash
./run_test_data_scripts.sh
```

### Manual Materialized View Refresh

To manually refresh the materialized views:
```sql
SELECT refresh_store_materialized_views();
```

## Configuration

The scripts use the database configuration from your `.env` file:
- `DATABASE_URL`: PostgreSQL connection string
- Database should have the platform schema

## Data Generation Limits

Configured in `test_data_creator.py`:
- 100 users
- 100 agent blocks
- 1-5 graphs per user
- 2-5 nodes per graph
- 1-5 presets per user
- 1-10 library agents per user
- 1-20 executions per graph
- 1-5 reviews per store listing version

## Notes

- All image URLs use `picsum.photos` for consistency with Next.js image configuration
- The scripts create realistic relationships between entities
- Materialized views are refreshed at the end of each script
- Data is designed to test both happy paths and edge cases
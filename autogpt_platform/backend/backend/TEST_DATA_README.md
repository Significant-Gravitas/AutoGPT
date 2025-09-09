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

### check_db.py
Tests and verifies materialized views functionality:
- Checks pg_cron job status (for automatic refresh)
- Displays current materialized view counts
- Adds test data (executions and reviews)
- Creates store listings if none exist
- Manually refreshes materialized views
- Compares before/after counts to verify updates
- Provides a summary of test results

## Materialized Views

The scripts test three key database views:

1. **mv_agent_run_counts**: Tracks execution counts by agent
2. **mv_review_stats**: Tracks review statistics (count, average rating) by store listing
3. **StoreAgent**: A view that combines store listing data with execution counts and ratings for display

The materialized views (mv_agent_run_counts and mv_review_stats) are automatically refreshed every 15 minutes via pg_cron, or can be manually refreshed using the `refresh_store_materialized_views()` function.

## Usage

### Prerequisites

1. Ensure the database is running:
```bash
docker compose up -d
# or for test database:
docker compose -f docker-compose.test.yaml --env-file ../.env up -d
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

#### Option 2: Run individually
```bash
# From backend/test directory:
# Create initial test data
poetry run python test_data_creator.py

# Update data to test materialized view changes
poetry run python test_data_updater.py

# From backend directory:
# Test materialized views functionality
poetry run python check_db.py

# Check store data status
poetry run python check_store_data.py
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

## Troubleshooting

### Reviews and StoreAgent view showing 0

If `check_db.py` shows that reviews remain at 0 and StoreAgent view shows 0 store agents:

1. **No store listings exist**: The script will automatically create test store listings if none exist
2. **No approved versions**: Store listings need approved versions to appear in the StoreAgent view
3. **Check with `check_store_data.py`**: This script provides detailed information about:
   - Total store listings
   - Store listing versions by status
   - Existing reviews
   - StoreAgent view contents
   - Agent graph executions

### pg_cron not installed

The warning "pg_cron extension is not installed" is normal in local development environments. The materialized views can still be refreshed manually using the `refresh_store_materialized_views()` function, which all scripts do automatically.

### Common Issues

- **Type errors with None values**: Fixed in the latest version of check_db.py by using `or 0` for nullable numeric fields
- **Missing relations**: Ensure you're using the correct field names (e.g., `StoreListing` not `storeListing` in includes)
- **Column name mismatches**: The database uses camelCase for column names (e.g., `agentGraphId` not `agent_graph_id`)
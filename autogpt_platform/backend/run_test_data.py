#!/usr/bin/env python3
"""
Run test data creation and update scripts in sequence.

Usage:
    poetry run python run_test_data.py
"""

import asyncio
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> bool:
    """Run a command and return True if successful."""
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, cwd=cwd
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return False


async def main():
    """Main function to run test data scripts."""
    print("=" * 60)
    print("Running Test Data Scripts for AutoGPT Platform")
    print("=" * 60)
    print()

    # Get the backend directory
    backend_dir = Path(__file__).parent
    test_dir = backend_dir / "test"

    # Check if we're in the right directory
    if not (backend_dir / "pyproject.toml").exists():
        print("ERROR: This script must be run from the backend directory")
        sys.exit(1)

    print("1. Checking database connection...")
    print("-" * 40)

    # Import here to ensure proper environment setup
    try:
        from prisma import Prisma

        db = Prisma()
        await db.connect()
        print("✓ Database connection successful")
        await db.disconnect()
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("\nPlease ensure:")
        print("1. The database services are running (docker compose up -d)")
        print("2. The DATABASE_URL in .env is correct")
        print("3. Migrations have been run (poetry run prisma migrate deploy)")
        sys.exit(1)

    print()
    print("2. Running test data creator...")
    print("-" * 40)

    # Run test_data_creator.py
    if run_command(["poetry", "run", "python", "test_data_creator.py"], cwd=test_dir):
        print()
        print("✅ Test data created successfully!")

        print()
        print("3. Running test data updater...")
        print("-" * 40)

        # Run test_data_updater.py
        if run_command(
            ["poetry", "run", "python", "test_data_updater.py"], cwd=test_dir
        ):
            print()
            print("✅ Test data updated successfully!")
        else:
            print()
            print("❌ Test data updater failed!")
            sys.exit(1)
    else:
        print()
        print("❌ Test data creator failed!")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Test data setup completed successfully!")
    print("=" * 60)
    print()
    print("The materialized views have been populated with test data:")
    print("- mv_agent_run_counts: Agent execution statistics")
    print("- mv_review_stats: Store listing review statistics")
    print()
    print("You can now:")
    print("1. Run tests: poetry run test")
    print("2. Start the backend: poetry run serve")
    print("3. View data in the database")
    print()


if __name__ == "__main__":
    asyncio.run(main())

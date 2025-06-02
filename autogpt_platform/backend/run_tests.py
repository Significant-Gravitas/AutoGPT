import os
import subprocess
import sys
import time


def wait_for_postgres(max_retries=5, delay=5):
    for _ in range(max_retries):
        try:
            result = subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    "docker-compose.test.yaml",
                    "exec",
                    "postgres-test",
                    "pg_isready",
                    "-U",
                    "postgres",
                    "-d",
                    "postgres",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if "accepting connections" in result.stdout:
                print("PostgreSQL is ready.")
                return True
        except subprocess.CalledProcessError:
            print(f"PostgreSQL is not ready yet. Retrying in {delay} seconds...")
            time.sleep(delay)
    print("Failed to connect to PostgreSQL.")
    return False


def run_command(command, check=True):
    try:
        subprocess.run(command, check=check)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        sys.exit(1)


def test():
    # Start PostgreSQL with Docker Compose
    run_command(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.test.yaml",
            "up",
            "-d",
        ]
    )

    if not wait_for_postgres():
        run_command(["docker", "compose", "-f", "docker-compose.test.yaml", "down"])
        sys.exit(1)

    # IMPORTANT: Set test database environment variables to prevent accidentally
    # resetting the developer's local database.
    #
    # This script spins up a separate test database container (postgres-test) using
    # docker-compose.test.yaml. We explicitly set DATABASE_URL and DIRECT_URL to point
    # to this test database to ensure that:
    # 1. The prisma migrate reset command only affects the test database
    # 2. Tests run against the test database, not the developer's local database
    # 3. Any database operations during testing are isolated from development data
    #
    # Without this, if a developer has DATABASE_URL set in their environment pointing
    # to their development database, running tests would wipe their local data!
    test_env = os.environ.copy()

    # Use environment variables if set, otherwise use defaults that match docker-compose.test.yaml
    db_user = os.getenv("DB_USER", "postgres")
    db_pass = os.getenv("DB_PASS", "postgres")
    db_name = os.getenv("DB_NAME", "postgres")
    db_port = os.getenv("DB_PORT", "5432")

    # Construct the test database URL - this ensures we're always pointing to the test container
    test_env["DATABASE_URL"] = (
        f"postgresql://{db_user}:{db_pass}@localhost:{db_port}/{db_name}"
    )
    test_env["DIRECT_URL"] = test_env["DATABASE_URL"]

    test_env["DB_PORT"] = db_port
    test_env["DB_NAME"] = db_name
    test_env["DB_PASS"] = db_pass
    test_env["DB_USER"] = db_user

    # Run Prisma migrations with test database
    # First, reset the database to ensure clean state for tests
    # This is safe because we've explicitly set DATABASE_URL to the test database above
    subprocess.run(
        ["prisma", "migrate", "reset", "--force", "--skip-seed"],
        env=test_env,
        check=False,
    )
    # Then apply migrations to get the test database schema up to date
    subprocess.run(["prisma", "migrate", "deploy"], env=test_env, check=True)

    # Run the tests with test database environment
    # This ensures all database connections in the tests use the test database,
    # not any database that might be configured in the developer's environment
    result = subprocess.run(["pytest"] + sys.argv[1:], env=test_env, check=False)

    run_command(["docker", "compose", "-f", "docker-compose.test.yaml", "down"])

    sys.exit(result.returncode)

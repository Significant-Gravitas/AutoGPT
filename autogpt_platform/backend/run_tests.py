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
            "postgres-test",
        ]
    )

    if not wait_for_postgres():
        run_command(["docker", "compose", "-f", "docker-compose.test.yaml", "down"])
        sys.exit(1)

    # Run Prisma migrations
    run_command(["prisma", "migrate", "dev"])

    # Run the tests
    result = subprocess.run(["pytest"] + sys.argv[1:], check=False)

    run_command(["docker", "compose", "-f", "docker-compose.test.yaml", "down"])

    sys.exit(result.returncode)

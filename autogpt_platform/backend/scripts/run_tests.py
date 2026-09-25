import os
import subprocess
import sys
import time

# How long a single readiness probe may take before it is abandoned.
# ``max_retries`` bounds the number of attempts, not their duration: without a
# timeout a wedged ``docker compose exec`` never returns, so the loop never
# advances and the runner hangs before it can retry or tear the stack down.
# ``docker compose exec`` has to start the compose CLI, parse the compose file
# and resolve the container before the probe itself runs, which on a loaded CI
# box is a few seconds on its own -- so ten seconds sits comfortably above a
# healthy round trip while staying well inside the budget of either retry loop
# (30 x 2s for Redis, 36 x 5s for Postgres). A single wedged probe then costs
# one attempt instead of the whole run.
PROBE_TIMEOUT_SECONDS = 10

# docker compose names the project after this directory, so every checkout's
# test stack is the same "backend" project with the same container names.
# An `up` from another worktree recreates the containers under a test session
# that is using them, and its `down` removes them. A run waits for another
# worktree's stack to go away, and only takes down a stack that is its own.
COMPOSE_PROJECT = "backend"
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACK_WAIT_SECONDS = 30 * 60


def wait_for_postgres(max_retries=36, delay=5):
    """Block until the `postgres` role can run a query.

    pg_isready isn't enough: on a fresh data directory the Supabase image runs
    its init scripts against a temporary server that already accepts
    connections, before the `postgres` role exists. Starting on that, or having
    the container recreated mid-init, leaves a half-built database that fails
    every later run with `role "postgres" does not exist`.
    """
    for _ in range(max_retries):
        try:
            result = subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    "docker-compose.test.yaml",
                    "--env-file",
                    "../.env",
                    "exec",
                    "db",
                    "psql",
                    "-U",
                    "postgres",
                    "-d",
                    "postgres",
                    "-tAc",
                    "select 1",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
            )
            if result.stdout.strip() == "1":
                print("PostgreSQL is ready.")
                return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        print(f"PostgreSQL is not ready yet. Retrying in {delay} seconds...")
        time.sleep(delay)
    print(
        "Failed to connect to PostgreSQL. If `docker compose -f "
        "docker-compose.test.yaml logs db` says the `postgres` role does not "
        "exist, the database's first start was interrupted: delete "
        "../db/docker/volumes/db/data and run the tests again."
    )
    return False


def wait_for_redis_cluster(max_retries=30, delay=2):
    """Block until the 3-shard cluster has finished forming.

    ``redis-init`` creates the cluster asynchronously after the shards come
    up. Until ``cluster_state`` is ``ok`` the backend's ``RedisCluster``
    client cannot resolve slots, and its connection retry backs off for tens
    of minutes rather than failing — so a test session started too early
    looks like a hang, not like a race.
    """
    for _ in range(max_retries):
        try:
            result = subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    "docker-compose.test.yaml",
                    "--env-file",
                    "../.env",
                    "exec",
                    "redis-0",
                    "redis-cli",
                    "-p",
                    "17000",
                    "cluster",
                    "info",
                ],
                check=False,  # readiness is read from stdout, not from the exit code
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
            )
            if "cluster_state:ok" in result.stdout:
                print("Redis cluster is ready.")
                return True
        except subprocess.TimeoutExpired:
            print(f"Redis cluster probe timed out after {PROBE_TIMEOUT_SECONDS}s.")
        print(f"Redis cluster is not ready yet. Retrying in {delay} seconds...")
        time.sleep(delay)
    print("Failed to form the Redis cluster.")
    return False


def run_command(command, check=True):
    try:
        subprocess.run(command, check=check)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        sys.exit(1)


def wait_for_stack_to_be_free(max_wait=STACK_WAIT_SECONDS, delay=10):
    """Wait until no other checkout's test run is using the test stack."""
    deadline = time.monotonic() + max_wait
    reported = set()
    while owners := other_stack_owners():
        in_use_by = ", ".join(sorted(owners))
        if time.monotonic() >= deadline:
            print(
                f"The test stack is still in use by {in_use_by}. If no test run "
                "is going there, take its stack down with `docker compose -f "
                "docker-compose.test.yaml down` from that directory."
            )
            return False
        if owners != reported:
            print(f"The test stack is in use by {in_use_by}; waiting for that run.")
            reported = owners
        time.sleep(delay)
    return True


def tear_down_stack():
    """Take the test stack down, unless another checkout has taken it over."""
    if owners := other_stack_owners():
        print(f"Leaving the test stack up for {', '.join(sorted(owners))}.")
        return
    run_command(["docker", "compose", "-f", "docker-compose.test.yaml", "down"])


def other_stack_owners():
    """Directories other than this one whose `docker compose` created the
    running containers of the test stack."""
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"label=com.docker.compose.project={COMPOSE_PROJECT}",
                "--format",
                '{{.Label "com.docker.compose.project.working_dir"}}',
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # Docker isn't answering; the compose command that follows will say so.
        print(f"`docker ps` timed out after {PROBE_TIMEOUT_SECONDS}s.")
        return set()
    owners = {_normalize_path(line) for line in result.stdout.splitlines() if line}
    return owners - {_normalize_path(BACKEND_DIR)}


def _normalize_path(path):
    return os.path.normcase(os.path.normpath(path.strip()))


def test():
    if not wait_for_stack_to_be_free():
        sys.exit(1)

    # Start PostgreSQL with Docker Compose
    run_command(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.test.yaml",
            "--env-file",
            "../.env",
            "up",
            "-d",
        ]
    )

    if not wait_for_postgres() or not wait_for_redis_cluster():
        tear_down_stack()
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

    # Load database configuration from .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), "../../.env")
    if os.path.exists(dotenv_path):
        with open(dotenv_path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

    # Get database config from environment (now populated from .env)
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_pass = os.getenv("POSTGRES_PASSWORD", "postgres")
    db_name = os.getenv("POSTGRES_DB", "postgres")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    # Run tests against a DEDICATED DATABASE on the test server. This is
    # load-bearing: the "test" db container shares its data directory with
    # the dev Supabase database, whose default search_path is
    # `"$user", platform, public` — so a schema-less URL points unqualified
    # DDL (and `prisma migrate reset --force`!) at the LIVE `platform`
    # schema, and a `?schema=` URL breaks migrations that rely on
    # extensions installed in Supabase's `extensions` schema (pg_trgm's
    # gin_trgm_ops). A separate database gets its own fresh `public`
    # schema: extensions install locally, resets stay contained.
    test_db_name = "agpt_test"
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker-compose.test.yaml",
            "--env-file",
            "../.env",
            "exec",
            "-T",
            "db",
            "psql",
            "-U",
            db_user,
            "-d",
            db_name,
            "-c",
            f"CREATE DATABASE {test_db_name}",
        ],
        check=False,  # already exists on reruns
        capture_output=True,
    )
    test_env["DATABASE_URL"] = (
        f"postgresql://{db_user}:{db_pass}@localhost:{db_port}/{test_db_name}"
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

    tear_down_stack()

    sys.exit(result.returncode)

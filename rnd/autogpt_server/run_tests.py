import subprocess
import sys


def test():
    # Start PostgreSQL with Docker Compose
    subprocess.run(["docker-compose", "up", "-d", "postgres"], check=True)

    result = subprocess.run(["pytest"] + sys.argv[1:], check=False)

    subprocess.run(["docker-compose", "down"], check=True)

    sys.exit(result.returncode)
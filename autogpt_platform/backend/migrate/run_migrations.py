import logging
import os
import subprocess

import psycopg2


def run_prisma_migrations():
    result = subprocess.run(
        ["prisma", "migrate", "deploy"], capture_output=True, text=True
    )
    if result.returncode != 0:
        logging.info("Prisma migration failed:")
        logging.error(result.stderr)
        return False
    return True


def run_custom_migrations():
    db_url = os.environ.get("DIRECT_DATABASE_URL")
    conn = psycopg2.connect(db_url)

    try:
        with conn.cursor() as cur:
            custom_migrations_dir = "./custom_migrations"
            for filename in sorted(os.listdir(custom_migrations_dir)):
                if filename.endswith(".sql"):
                    with open(os.path.join(custom_migrations_dir, filename), "r") as f:
                        sql = f.read()
                        cur.execute(sql)
                    logging.info(f"Executed custom migration: {filename}")
            conn.commit()
    except Exception as e:
        logging.exception(f"Custom migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

def main():
    logging.info("Starting migrations")
    if run_prisma_migrations():
        run_custom_migrations()
    logging.info("Migrations completed")

if __name__ == "__main__":
    run_prisma_migrations()
    run_custom_migrations()

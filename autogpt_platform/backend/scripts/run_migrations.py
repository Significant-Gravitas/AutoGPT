import subprocess
import os
import psycopg2


def run_prisma_migrations():
    result = subprocess.run(['prisma', 'migrate', 'deploy'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Prisma migration failed:")
        print(result.stderr)
        return False
    return True


def run_custom_migrations():
    db_url = os.environ.get('DATABASE_URL')
    conn = psycopg2.connect(db_url)

    try:
        with conn.cursor() as cur:
            custom_migrations_dir = './custom_migrations'
            for filename in sorted(os.listdir(custom_migrations_dir)):
                if filename.endswith('.sql'):
                    with open(os.path.join(custom_migrations_dir, filename), 'r') as f:
                        sql = f.read()
                        cur.execute(sql)
                    print(f"Executed custom migration: {filename}")
            conn.commit()
    except Exception as e:
        print(f"Custom migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

def migrate():
    run_prisma_migrations()
    run_custom_migrations()
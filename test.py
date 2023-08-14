DATABASE_NAME = "agent.db"
import sqlite3

# Read all data from database


def read_all():
    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks")
    rows = cur.fetchall()
    conn.close()
    return rows


print(read_all())

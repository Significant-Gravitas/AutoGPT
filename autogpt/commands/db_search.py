import json
from typing import Tuple, List, Dict, Any
import psycopg2
from unidecode import unidecode
from psycopg2.extras import RealDictCursor

from autogpt.config.external_db_config import EXTERNAL_DATABASE


def available_databases() -> list[dict[str, Any]]:
    """
    A function that return all available databases with description.

    """
    return [{'database name': key, 'description': value.get('description')} for key, value in
            EXTERNAL_DATABASE.items()]


def sql_query(database_name: str, sql_query: str) -> str:
    """
    A function that takes in a sql query as a string and returns a string response from a PostgresSql database.

    Parameters:
        sql_query (str): Sql query to be evaluated.
    Returns:
        A result string from database_query. A database response from a query passed in parameters.
    """
    return database_query(database_name, sql_query)


def database_table_names(database_name: str) -> str:
    """
    A function that return a string containing all table name of database_name.
    """
    return database_query(database_name,
                          """SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';""")


def database_table_column_names(database_name: str, table: str) -> str:
    """
    A function that return column table names according to table name
    """
    return database_query(database_name,
                          f"""SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = '{table}';""")


def database_query(database_name: str, sql_query: str) -> str:
    """
    A function that takes in a string and returns a string response from a PostgresSql database.

    Parameters:
        sql_query (str): Sql query to be evaluated.
    Returns:
        A result string from database_query. A database response from a query passed in parameters.
    """
    database = EXTERNAL_DATABASE.get(database_name)
    if database:
        conn = psycopg2.connect(**database.get('database'))
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Use a list here to insert query parameters into the query string.
        cur.execute(
            sql_query
        )

        result = unidecode(json.dumps(cur.fetchall(), ensure_ascii=False).encode('utf-8').decode('utf-8'))

        cur.close()

        return result

    return "Database does not exist"


if __name__ == '__main__':
    print(available_databases())
    # for key, value in EXTERNAL_DATABASE.items():
    #     print(key, value)

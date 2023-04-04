import json
from sqlalchemy import text, create_engine, MetaData, Table, Column, String, Integer, DateTime
from sqlalchemy.pool import NullPool
from config import Config
from llm_utils import create_chat_completion
import decimal, datetime

def alchemyencoder(obj):
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, decimal.Decimal):
        return float(obj)

def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = str(getattr(row, column.name))

    return d

cfg = Config()

config = None
db_config = None
engine = None
metadata = None

def init_db():
    global config
    global engine
    global metadata
    db_config = "mysql+pymysql://%(username)s:%(password)s@%(host)s:%(port)s/%(database)s?charset=utf8" % cfg.db
    engine = create_engine(db_config, poolclass=NullPool)
    metadata = MetaData()
    metadata.reflect(bind=engine)

def read_table(table_name):
    with engine.connect() as connection:
        result = connection.execute(text("SHOW COLUMNS FROM %(table)s;" % {'table': table_name})).fetchall()
        return json.dumps([list(r) for r in result])

def execute_sql(sql):
    with engine.connect() as connection:
        result = connection.execute(text(sql)).fetchall()
        return json.dumps([list(r) for r in result], default=alchemyencoder)

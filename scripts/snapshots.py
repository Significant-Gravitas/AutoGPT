import time
from config import Config
import data_store as ds

cfg = Config()

def create_snapshot():
    if not cfg.snapshots_enabled:
        return { "message_history": True, "memory": True }
    current_unix_time = int(time.time())
    message_history = ds.instance.persist_message_history(current_unix_time)
    memory = ds.instance.persist_memory(current_unix_time)
    return { "message_history": message_history, "memory": memory }

def load_snapshot(id):
    message_history = ds.instance.load_message_history(id)
    memory = ds.instance.load_memory(id)
    return { "message_history": message_history, "memory": memory }

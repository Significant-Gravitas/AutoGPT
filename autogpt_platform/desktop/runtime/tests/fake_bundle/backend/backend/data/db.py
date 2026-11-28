from backend.services import marked


def is_connected() -> bool:
    return marked("database.connected")

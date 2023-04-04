import snapshots

message_history = []

def append(message):
    message_history.append(message)
    result = snapshots.create_snapshot()
    if result["message_history"] is not True:
        print("Failed to persist message history")
        print(result["message_history"])
    return message

def overwrite(key, message):
    message_history[key] = message
    result = snapshots.create_snapshot()
    if result["message_history"] is not True:
        print("Failed to persist message history")
        print(result["message_history"])
    return message

def set_history(history):
    global message_history
    message_history = history

def clear_history():
    global message_history
    message_history = []
    return True

import snapshots

permanent_memory = []

def append(string):
    permanent_memory.append(string)
    result = snapshots.create_snapshot()
    if result["memory"] is not True:
        print("Failed to persist memory")
        print(result["memory"])
    return True

def delete(key):
    if key >= 0 and key < len(permanent_memory):
        del permanent_memory[key]
        result = snapshots.create_snapshot()
        if result["memory"] is not True:
            print("Failed to persist memory")
            print(result["memory"])
        return key
    else:
        return None

def commit_memory(string):
    permanent_memory.append(string)
    result = snapshots.create_snapshot()
    if result["memory"] is not True:
        print("Failed to persist memory")
        print(result["memory"])
    return True

def delete_memory(key):
    if key in permanent_memory:
        del permanent_memory[key]
        result = snapshots.create_snapshot()
        if result["memory"] is not True:
            print("Failed to persist memory")
            print(result["memory"])
        return key
    else:
        return None

def overwrite_memory(key, string):
    if key in permanent_memory:
        permanent_memory[key] = string
        result = snapshots.create_snapshot()
        if result["memory"] is not True:
            print("Failed to persist memory")
            print(result["memory"])
        return key
    else:
        return None

def clear_memory():
    permanent_memory.clear()
    return True

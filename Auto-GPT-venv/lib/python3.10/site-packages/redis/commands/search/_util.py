def to_string(s):
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode("utf-8", "ignore")
    else:
        return s  # Not a string we care about

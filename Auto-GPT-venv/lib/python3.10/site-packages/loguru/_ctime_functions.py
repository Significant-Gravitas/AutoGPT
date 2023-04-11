import os


def load_ctime_functions():
    if os.name == "nt":
        import win32_setctime

        def get_ctime_windows(filepath):
            return os.stat(filepath).st_ctime

        def set_ctime_windows(filepath, timestamp):
            if not win32_setctime.SUPPORTED:
                return

            try:
                win32_setctime.setctime(filepath, timestamp)
            except (OSError, ValueError):
                pass

        return get_ctime_windows, set_ctime_windows

    elif hasattr(os.stat_result, "st_birthtime"):

        def get_ctime_macos(filepath):
            return os.stat(filepath).st_birthtime

        def set_ctime_macos(filepath, timestamp):
            pass

        return get_ctime_macos, set_ctime_macos

    elif hasattr(os, "getxattr") and hasattr(os, "setxattr"):

        def get_ctime_linux(filepath):
            try:
                return float(os.getxattr(filepath, b"user.loguru_crtime"))
            except OSError:
                return os.stat(filepath).st_mtime

        def set_ctime_linux(filepath, timestamp):
            try:
                os.setxattr(filepath, b"user.loguru_crtime", str(timestamp).encode("ascii"))
            except OSError:
                pass

        return get_ctime_linux, set_ctime_linux

    def get_ctime_fallback(filepath):
        return os.stat(filepath).st_mtime

    def set_ctime_fallback(filepath, timestamp):
        pass

    return get_ctime_fallback, set_ctime_fallback


get_ctime, set_ctime = load_ctime_functions()

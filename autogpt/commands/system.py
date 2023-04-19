import platform

def get_current_os() -> str:
    """Return the current operating system and its version

    Returns:
        str: The current operating system and its version
    """
    os_name = platform.system()
    os_version = platform.release()
    return f"Current operating system: {os_name} {os_version}"
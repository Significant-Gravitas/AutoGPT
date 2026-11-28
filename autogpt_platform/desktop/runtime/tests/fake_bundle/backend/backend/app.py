def run_processes(*processes, **kwargs):
    """As upstream's: the last process runs in the foreground."""
    processes[-1].start(background=False, **kwargs)

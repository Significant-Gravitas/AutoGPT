def raise_with_traceback(exc_type, traceback, *args, **kwargs):
    """
    Raise a new exception of type `exc_type` with an existing `traceback`. All
    additional (keyword-)arguments are forwarded to `exc_type`
    """
    raise exc_type(*args, **kwargs), None, traceback

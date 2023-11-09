def open_in_browser(html):
    """
    Open the HTML document in a web browser, saving it to a temporary
    file to open it.  Note that this does not delete the file after
    use.  This is mainly meant for debugging.
    """
    import os
    import webbrowser
    import tempfile

    handle, fn = tempfile.mkstemp(suffix=".html")
    f = os.fdopen(handle, "wb")
    try:
        f.write(b"<meta charset='UTF-8' />")
        f.write(html.encode("utf-8"))
    finally:
        # we leak the file itself here, but we should at least close it
        f.close()
    url = "file://" + fn.replace(os.path.sep, "/")
    webbrowser.open(url)
    return url

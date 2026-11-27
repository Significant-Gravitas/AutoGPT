TEST_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8"
    "AAAAASUVORK5CYII="
)

TEST_HTML = "<html><body>Example page content</body></html>"


def stored_screenshot(screenshot: str, execution_context: object) -> str:
    return f"data:image/png;base64,{screenshot}"

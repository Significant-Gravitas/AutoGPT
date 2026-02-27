"""Browser Agent - Headless browser automation powered by agent-browser CLI.

Provides interactive browser capabilities beyond basic HTTP scraping:
- Navigate to URLs and interact with JavaScript-rendered pages
- Click elements, type text, fill forms
- Take screenshots for visual analysis
- Get accessibility tree snapshots optimized for AI consumption
- Persistent browser sessions for multi-step workflows (logins, forms, etc.)

Requires: agent-browser CLI (npm install -g agent-browser)
"""

import json
import os
import subprocess
import shutil
from config import Config

cfg = Config()

# Directory to store browser screenshots and artifacts
BROWSER_WORKSPACE = os.path.join(os.path.dirname(__file__), '..', 'auto_gpt_workspace', 'browser')


def _ensure_workspace():
    """Ensure the browser workspace directory exists."""
    os.makedirs(BROWSER_WORKSPACE, exist_ok=True)


def _get_agent_browser_path():
    """Find the agent-browser CLI binary."""
    path = shutil.which("agent-browser")
    if path:
        return path
    # Check common npx-accessible locations
    npx_path = shutil.which("npx")
    if npx_path:
        return None  # Signal to use npx fallback
    return None


def _run_command(args, timeout=60):
    """Run an agent-browser CLI command and return the output.

    Args:
        args: List of command arguments (excluding the 'agent-browser' prefix).
        timeout: Max seconds to wait for the command to complete.

    Returns:
        A string with the command output, or an error message.
    """
    agent_browser = _get_agent_browser_path()

    if agent_browser:
        cmd = [agent_browser] + args
    else:
        # Fall back to npx
        cmd = ["npx", "agent-browser"] + args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=BROWSER_WORKSPACE,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr:
                return f"Error: {stderr}"
            return f"Error: agent-browser exited with code {result.returncode}"
        return result.stdout.strip()
    except FileNotFoundError:
        return (
            "Error: agent-browser CLI not found. "
            "Install it with: npm install -g agent-browser"
        )
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error: {str(e)}"


def browser_navigate(url):
    """Navigate the browser to a URL.

    Args:
        url: The URL to navigate to.

    Returns:
        Page title and status, or an error message.
    """
    _ensure_workspace()
    return _run_command(["navigate", url])


def browser_snapshot():
    """Get an accessibility tree snapshot of the current page.

    Returns a structured representation of the page content optimized for
    AI consumption — includes text, roles, labels, and interactive elements.

    Returns:
        The accessibility snapshot as a string.
    """
    _ensure_workspace()
    return _run_command(["snapshot"])


def browser_screenshot(filename="screenshot.png"):
    """Take a screenshot of the current page.

    Args:
        filename: Name for the screenshot file (saved in browser workspace).

    Returns:
        Path to the saved screenshot, or an error message.
    """
    _ensure_workspace()
    filepath = os.path.join(BROWSER_WORKSPACE, filename)
    result = _run_command(["screenshot", "--output", filepath])
    if result.startswith("Error"):
        return result
    return f"Screenshot saved to: {filepath}"


def browser_click(selector):
    """Click an element on the current page.

    Args:
        selector: CSS selector or semantic locator (e.g., 'button:has-text("Submit")').

    Returns:
        Result of the click action, or an error message.
    """
    _ensure_workspace()
    return _run_command(["click", selector])


def browser_type(selector, text):
    """Type text into an input element on the current page.

    Args:
        selector: CSS selector or semantic locator for the input element.
        text: The text to type.

    Returns:
        Result of the type action, or an error message.
    """
    _ensure_workspace()
    return _run_command(["type", selector, text])


def browser_fill(selector, value):
    """Fill a form field with a value (clears existing content first).

    Args:
        selector: CSS selector or semantic locator for the form field.
        value: The value to fill in.

    Returns:
        Result of the fill action, or an error message.
    """
    _ensure_workspace()
    return _run_command(["fill", selector, value])


def browser_hover(selector):
    """Hover over an element on the current page.

    Args:
        selector: CSS selector or semantic locator for the element.

    Returns:
        Result of the hover action, or an error message.
    """
    _ensure_workspace()
    return _run_command(["hover", selector])


def browser_scroll(direction="down", amount="page"):
    """Scroll the current page.

    Args:
        direction: Scroll direction — 'up' or 'down'.
        amount: How much to scroll — 'page' for a full page, or a pixel value.

    Returns:
        Result of the scroll action, or an error message.
    """
    _ensure_workspace()
    return _run_command(["scroll", f"--direction={direction}", f"--amount={amount}"])


def browser_get_text(selector):
    """Get the text content of an element.

    Args:
        selector: CSS selector or semantic locator for the element.

    Returns:
        The text content of the element, or an error message.
    """
    _ensure_workspace()
    return _run_command(["text", selector])

"""
Server-rendered HTML templates for OAuth consent UI.

These templates are used for the OAuth authorization flow
when the user needs to approve access for an external application.
"""

from typing import Optional

from backend.server.oauth.models import SCOPE_DESCRIPTIONS


def _base_styles() -> str:
    """Common CSS styles for all OAuth pages."""
    return """
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            color: #e4e4e7;
        }
        .container {
            background: #27272a;
            border-radius: 16px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            max-width: 420px;
            width: 100%;
            padding: 32px;
        }
        .header {
            text-align: center;
            margin-bottom: 24px;
        }
        .logo {
            width: 64px;
            height: 64px;
            border-radius: 12px;
            margin-bottom: 16px;
            background: #3f3f46;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-left: auto;
            margin-right: auto;
        }
        .logo img {
            max-width: 48px;
            max-height: 48px;
            border-radius: 8px;
        }
        .logo-placeholder {
            font-size: 28px;
            color: #a1a1aa;
        }
        h1 {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .subtitle {
            color: #a1a1aa;
            font-size: 14px;
        }
        .app-name {
            color: #22d3ee;
            font-weight: 600;
        }
        .divider {
            height: 1px;
            background: #3f3f46;
            margin: 24px 0;
        }
        .scopes-section h2 {
            font-size: 14px;
            font-weight: 500;
            color: #a1a1aa;
            margin-bottom: 16px;
        }
        .scope-item {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 12px 0;
            border-bottom: 1px solid #3f3f46;
        }
        .scope-item:last-child {
            border-bottom: none;
        }
        .scope-icon {
            width: 20px;
            height: 20px;
            color: #22d3ee;
            flex-shrink: 0;
            margin-top: 2px;
        }
        .scope-text {
            font-size: 14px;
            line-height: 1.5;
        }
        .buttons {
            display: flex;
            gap: 12px;
            margin-top: 24px;
        }
        .btn {
            flex: 1;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border: none;
            transition: all 0.2s;
        }
        .btn-cancel {
            background: #3f3f46;
            color: #e4e4e7;
        }
        .btn-cancel:hover {
            background: #52525b;
        }
        .btn-allow {
            background: #22d3ee;
            color: #0f172a;
        }
        .btn-allow:hover {
            background: #06b6d4;
        }
        .footer {
            margin-top: 24px;
            text-align: center;
            font-size: 12px;
            color: #71717a;
        }
        .footer a {
            color: #a1a1aa;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        .error-container {
            text-align: center;
        }
        .error-icon {
            width: 64px;
            height: 64px;
            margin: 0 auto 16px;
            color: #ef4444;
        }
        .error-title {
            color: #ef4444;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .error-message {
            color: #a1a1aa;
            font-size: 14px;
            margin-bottom: 24px;
        }
        .success-icon {
            width: 64px;
            height: 64px;
            margin: 0 auto 16px;
            color: #22c55e;
        }
        .success-title {
            color: #22c55e;
        }
    """


def _check_icon() -> str:
    """SVG checkmark icon."""
    return """
        <svg class="scope-icon" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
        </svg>
    """


def _error_icon() -> str:
    """SVG error icon."""
    return """
        <svg class="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
    """


def _success_icon() -> str:
    """SVG success icon."""
    return """
        <svg class="success-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="M9 12l2 2 4-4"/>
        </svg>
    """


def render_consent_page(
    client_name: str,
    client_logo: Optional[str],
    scopes: list[str],
    consent_token: str,
    action_url: str,
    privacy_policy_url: Optional[str] = None,
    terms_url: Optional[str] = None,
) -> str:
    """
    Render the OAuth consent page.

    Args:
        client_name: Name of the requesting application
        client_logo: URL to the client's logo (optional)
        scopes: List of requested scopes
        consent_token: CSRF token for the consent form
        action_url: URL to submit the consent form
        privacy_policy_url: Client's privacy policy URL (optional)
        terms_url: Client's terms of service URL (optional)

    Returns:
        HTML string for the consent page
    """
    # Build logo HTML
    if client_logo:
        logo_html = f'<img src="{client_logo}" alt="{client_name}">'
    else:
        logo_html = f'<span class="logo-placeholder">{client_name[0].upper()}</span>'

    # Build scopes HTML
    scopes_html = ""
    for scope in scopes:
        description = SCOPE_DESCRIPTIONS.get(scope, scope)
        scopes_html += f"""
            <div class="scope-item">
                {_check_icon()}
                <span class="scope-text">{description}</span>
            </div>
        """

    # Build footer links
    footer_links = []
    if privacy_policy_url:
        footer_links.append(
            f'<a href="{privacy_policy_url}" target="_blank">Privacy Policy</a>'
        )
    if terms_url:
        footer_links.append(
            f'<a href="{terms_url}" target="_blank">Terms of Service</a>'
        )
    footer_html = " &bull; ".join(footer_links) if footer_links else ""

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Authorize {client_name} - AutoGPT</title>
        <style>{_base_styles()}</style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">{logo_html}</div>
                <h1>Authorize <span class="app-name">{client_name}</span></h1>
                <p class="subtitle">wants to access your AutoGPT account</p>
            </div>

            <div class="divider"></div>

            <div class="scopes-section">
                <h2>This will allow {client_name} to:</h2>
                {scopes_html}
            </div>

            <form method="POST" action="{action_url}">
                <input type="hidden" name="consent_token" value="{consent_token}">
                <div class="buttons">
                    <button type="submit" name="authorize" value="false" class="btn btn-cancel">
                        Cancel
                    </button>
                    <button type="submit" name="authorize" value="true" class="btn btn-allow">
                        Allow
                    </button>
                </div>
            </form>

            {f'<div class="footer">{footer_html}</div>' if footer_html else ''}
        </div>
    </body>
    </html>
    """


def render_error_page(
    error: str,
    error_description: str,
    redirect_url: Optional[str] = None,
) -> str:
    """
    Render an OAuth error page.

    Args:
        error: Error code
        error_description: Human-readable error description
        redirect_url: Optional URL to redirect back (if safe)

    Returns:
        HTML string for the error page
    """
    redirect_html = ""
    if redirect_url:
        redirect_html = f"""
            <a href="{redirect_url}" class="btn btn-cancel" style="display: inline-block; text-decoration: none;">
                Go Back
            </a>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Authorization Error - AutoGPT</title>
        <style>{_base_styles()}</style>
    </head>
    <body>
        <div class="container">
            <div class="error-container">
                {_error_icon()}
                <h1 class="error-title">Authorization Failed</h1>
                <p class="error-message">{error_description}</p>
                <p class="error-message" style="font-size: 12px; color: #52525b;">
                    Error code: {error}
                </p>
                {redirect_html}
            </div>
        </div>
    </body>
    </html>
    """


def render_success_page(
    message: str,
    redirect_origin: Optional[str] = None,
    post_message_data: Optional[dict] = None,
) -> str:
    """
    Render a success page, optionally with postMessage for popup flows.

    Args:
        message: Success message to display
        redirect_origin: Origin for postMessage (popup flows)
        post_message_data: Data to send via postMessage (popup flows)

    Returns:
        HTML string for the success page
    """
    # PostMessage script for popup flows
    post_message_script = ""
    if redirect_origin and post_message_data:
        import json

        post_message_script = f"""
            <script>
                (function() {{
                    var targetOrigin = {json.dumps(redirect_origin)};
                    var message = {json.dumps(post_message_data)};
                    if (window.opener) {{
                        window.opener.postMessage(message, targetOrigin);
                        setTimeout(function() {{ window.close(); }}, 1000);
                    }}
                }})();
            </script>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Authorization Successful - AutoGPT</title>
        <style>{_base_styles()}</style>
    </head>
    <body>
        <div class="container">
            <div class="error-container">
                {_success_icon()}
                <h1 class="success-title">Success!</h1>
                <p class="error-message">{message}</p>
                <p class="error-message" style="font-size: 12px;">
                    This window will close automatically...
                </p>
            </div>
        </div>
        {post_message_script}
    </body>
    </html>
    """


def render_login_redirect_page(login_url: str) -> str:
    """
    Render a page that redirects to login.

    Args:
        login_url: URL to redirect to for login

    Returns:
        HTML string with auto-redirect
    """
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="refresh" content="0;url={login_url}">
        <title>Login Required - AutoGPT</title>
        <style>{_base_styles()}</style>
    </head>
    <body>
        <div class="container">
            <div class="error-container">
                <p class="error-message">Redirecting to login...</p>
                <a href="{login_url}" class="btn btn-allow" style="display: inline-block; text-decoration: none;">
                    Click here if not redirected
                </a>
            </div>
        </div>
    </body>
    </html>
    """

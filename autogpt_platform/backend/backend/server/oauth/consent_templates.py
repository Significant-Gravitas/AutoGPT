"""
Server-rendered HTML templates for OAuth consent UI.

These templates are used for the OAuth authorization flow
when the user needs to approve access for an external application.
"""

import html
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
    # Escape user-provided values to prevent XSS
    safe_client_name = html.escape(client_name)
    safe_client_logo = html.escape(client_logo) if client_logo else None

    # Build logo HTML
    if safe_client_logo:
        logo_html = f'<img src="{safe_client_logo}" alt="{safe_client_name}">'
    else:
        logo_html = f'<span class="logo-placeholder">{html.escape(client_name[0].upper())}</span>'

    # Build scopes HTML
    scopes_html = ""
    for scope in scopes:
        description = SCOPE_DESCRIPTIONS.get(scope, scope)
        scopes_html += f"""
            <div class="scope-item">
                {_check_icon()}
                <span class="scope-text">{html.escape(description)}</span>
            </div>
        """

    # Build footer links (escape URLs)
    footer_links = []
    if privacy_policy_url:
        footer_links.append(
            f'<a href="{html.escape(privacy_policy_url)}" target="_blank">Privacy Policy</a>'
        )
    if terms_url:
        footer_links.append(
            f'<a href="{html.escape(terms_url)}" target="_blank">Terms of Service</a>'
        )
    footer_html = " &bull; ".join(footer_links) if footer_links else ""

    # Escape action_url and consent_token
    safe_action_url = html.escape(action_url)
    safe_consent_token = html.escape(consent_token)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Authorize {safe_client_name} - AutoGPT</title>
        <style>{_base_styles()}</style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">{logo_html}</div>
                <h1>Authorize <span class="app-name">{safe_client_name}</span></h1>
                <p class="subtitle">wants to access your AutoGPT account</p>
            </div>

            <div class="divider"></div>

            <div class="scopes-section">
                <h2>This will allow {safe_client_name} to:</h2>
                {scopes_html}
            </div>

            <form method="POST" action="{safe_action_url}">
                <input type="hidden" name="consent_token" value="{safe_consent_token}">
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
    # Escape user-provided values to prevent XSS
    safe_error = html.escape(error)
    safe_error_description = html.escape(error_description)

    redirect_html = ""
    if redirect_url:
        safe_redirect_url = html.escape(redirect_url)
        redirect_html = f"""
            <a href="{safe_redirect_url}" class="btn btn-cancel" style="display: inline-block; text-decoration: none;">
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
                <p class="error-message">{safe_error_description}</p>
                <p class="error-message" style="font-size: 12px; color: #52525b;">
                    Error code: {safe_error}
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
    # Escape user-provided values to prevent XSS
    safe_message = html.escape(message)

    # PostMessage script for popup flows
    post_message_script = ""
    if redirect_origin and post_message_data:
        import json

        # json.dumps escapes for JS context, but we also escape < > for HTML context
        safe_json_origin = (
            json.dumps(redirect_origin).replace("<", "\\u003c").replace(">", "\\u003e")
        )
        safe_json_data = (
            json.dumps(post_message_data)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )

        post_message_script = f"""
            <script>
                (function() {{
                    var targetOrigin = {safe_json_origin};
                    var message = {safe_json_data};
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
                <p class="error-message">{safe_message}</p>
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
    # Escape URL to prevent XSS
    safe_login_url = html.escape(login_url)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="refresh" content="0;url={safe_login_url}">
        <title>Login Required - AutoGPT</title>
        <style>{_base_styles()}</style>
    </head>
    <body>
        <div class="container">
            <div class="error-container">
                <p class="error-message">Redirecting to login...</p>
                <a href="{safe_login_url}" class="btn btn-allow" style="display: inline-block; text-decoration: none;">
                    Click here if not redirected
                </a>
            </div>
        </div>
    </body>
    </html>
    """


def _login_form_styles() -> str:
    """Additional CSS styles for login form."""
    return """
        .form-group {
            margin-bottom: 16px;
        }
        .form-group label {
            display: block;
            font-size: 14px;
            font-weight: 500;
            color: #a1a1aa;
            margin-bottom: 8px;
        }
        .form-group input {
            width: 100%;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #3f3f46;
            background: #18181b;
            color: #e4e4e7;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        .form-group input:focus {
            border-color: #22d3ee;
        }
        .form-group input::placeholder {
            color: #52525b;
        }
        .error-alert {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
            color: #fca5a5;
            font-size: 14px;
        }
        .btn-login {
            width: 100%;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border: none;
            background: #22d3ee;
            color: #0f172a;
            transition: all 0.2s;
            margin-top: 8px;
        }
        .btn-login:hover {
            background: #06b6d4;
        }
        .btn-login:disabled {
            background: #3f3f46;
            color: #71717a;
            cursor: not-allowed;
        }
        .signup-link {
            text-align: center;
            margin-top: 16px;
            font-size: 14px;
            color: #a1a1aa;
        }
        .signup-link a {
            color: #22d3ee;
            text-decoration: none;
        }
        .signup-link a:hover {
            text-decoration: underline;
        }
    """


def render_login_page(
    action_url: str,
    login_state: str,
    client_name: Optional[str] = None,
    error_message: Optional[str] = None,
    signup_url: Optional[str] = None,
    browser_login_url: Optional[str] = None,
) -> str:
    """
    Render an embedded login page for OAuth flow.

    Args:
        action_url: URL to submit the login form
        login_state: State token to preserve OAuth parameters
        client_name: Name of the application requesting access (optional)
        error_message: Error message to display (optional)
        signup_url: URL to signup page (optional)
        browser_login_url: URL to redirect to frontend login (optional)

    Returns:
        HTML string for the login page
    """
    # Escape all user-provided values to prevent XSS
    safe_action_url = html.escape(action_url)
    safe_login_state = html.escape(login_state)
    safe_client_name = html.escape(client_name) if client_name else None

    error_html = ""
    if error_message:
        safe_error_message = html.escape(error_message)
        error_html = f'<div class="error-alert">{safe_error_message}</div>'

    subtitle = "wants to access your AutoGPT account" if safe_client_name else ""
    title_html = (
        '<h1>Sign in to <span class="app-name">AutoGPT</span></h1>'
        if not safe_client_name
        else f'<h1><span class="app-name">{safe_client_name}</span></h1>'
    )

    signup_html = ""
    if signup_url:
        safe_signup_url = html.escape(signup_url)
        signup_html = f"""
            <div class="signup-link">
                Don't have an account? <a href="{safe_signup_url}">Sign up</a>
            </div>
        """

    browser_login_html = ""
    if browser_login_url:
        safe_browser_login_url = html.escape(browser_login_url)
        browser_login_html = f"""
            <div class="divider"></div>
            <div class="signup-link">
                <a href="{safe_browser_login_url}">Sign in with Google or other providers</a>
            </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sign In - AutoGPT</title>
        <style>
            {_base_styles()}
            {_login_form_styles()}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <span class="logo-placeholder">A</span>
                </div>
                {title_html}
                <p class="subtitle">{subtitle}</p>
            </div>

            <div class="divider"></div>

            {error_html}

            <form method="POST" action="{safe_action_url}">
                <input type="hidden" name="login_state" value="{safe_login_state}">

                <div class="form-group">
                    <label for="email">Email</label>
                    <input
                        type="email"
                        id="email"
                        name="email"
                        placeholder="you@example.com"
                        required
                        autocomplete="email"
                    >
                </div>

                <div class="form-group">
                    <label for="password">Password</label>
                    <input
                        type="password"
                        id="password"
                        name="password"
                        placeholder="Enter your password"
                        required
                        autocomplete="current-password"
                    >
                </div>

                <button type="submit" class="btn-login">Sign In</button>
            </form>

            {signup_html}
            {browser_login_html}
        </div>
    </body>
    </html>
    """

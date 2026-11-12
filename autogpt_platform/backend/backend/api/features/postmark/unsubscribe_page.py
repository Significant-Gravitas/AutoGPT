"""The pages a person sees after clicking Unsubscribe in an email."""

import html
import secrets
from string import Template

from fastapi.responses import HTMLResponse

from backend.util.settings import Settings

settings = Settings()

_STILL_SENT = "Emails about your subscription and payments still reach you."


def confirm_page() -> HTMLResponse:
    # No action: the form posts back to this page's own URL, token included.
    return _page(
        "Unsubscribe from AutoGPT emails?",
        f"<p>You'll stop getting product emails from AutoGPT, such as your "
        f"Briefing and alerts. {_STILL_SENT}</p>"
        '<form method="post"><button type="submit">Unsubscribe</button></form>'
        f'<p class="aside">Would you rather get fewer? <a href="{_settings_url()}">'
        "Choose which emails you get</a>.</p>",
    )


def unsubscribed_page() -> HTMLResponse:
    return _page(
        "You're unsubscribed",
        f"<p>We won't send you product emails any more. {_STILL_SENT}</p>"
        f'<p class="aside">Changed your mind? <a href="{_settings_url()}">'
        "Turn emails back on</a> in your settings.</p>",
    )


def invalid_link_page() -> HTMLResponse:
    return _page(
        "This link doesn't work",
        "<p>It may be incomplete or out of date.</p>"
        f'<p class="aside">You can turn emails off in <a href="{_settings_url()}">'
        "your settings</a> instead.</p>",
        status_code=400,
    )


def _page(title: str, body: str, status_code: int = 200) -> HTMLResponse:
    nonce = secrets.token_urlsafe(18)
    page = _SHELL.substitute(title=html.escape(title), body=body, nonce=nonce)
    return HTMLResponse(page, status_code=status_code, headers=_headers(nonce))


def _settings_url() -> str:
    base = settings.config.frontend_base_url or settings.config.platform_base_url
    # /settings/account shows the email toggles only behind settings-notifications.
    return html.escape(f"{base}/profile/settings")


def _headers(nonce: str) -> dict[str, str]:
    return {
        "Content-Security-Policy": (
            f"default-src 'none'; style-src 'nonce-{nonce}'; "
            "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        ),
        "X-Robots-Tag": "noindex, nofollow",
    }


_SHELL = Template(
    """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="referrer" content="no-referrer">
    <title>$title</title>
    <style nonce="$nonce">
      :root { color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
      body { margin: 0; min-height: 100vh; display: grid; place-items: center;
        background: #f7f7f8; color: #202123; }
      main { width: min(440px, calc(100vw - 32px)); box-sizing: border-box; padding: 32px;
        border: 1px solid #dedee3; border-radius: 16px; background: #fff; }
      h1 { margin: 0 0 12px; font-size: 22px; line-height: 1.3; }
      p { margin: 0 0 20px; color: #4b4c55; line-height: 1.5; }
      p.aside { margin: 20px 0 0; font-size: 14px; }
      a { color: inherit; }
      button { width: 100%; min-height: 44px; border: 0; border-radius: 10px;
        background: #202123; color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
      @media (prefers-color-scheme: dark) {
        body { background: #17171a; color: #ececf1; }
        main { background: #202124; border-color: #34353a; }
        p { color: #b4b5bd; }
        button { background: #ececf1; color: #17171a; }
      }
    </style>
  </head>
  <body>
    <main>
      <h1>$title</h1>
      $body
    </main>
  </body>
</html>
"""
)

"""The private checkout's process-level configuration.

Deliberately read from the environment rather than ``Settings``: the payment
worker is spawned with a scrubbed environment, and the broker and egress proxy
run in their own containers, so none of them load the backend's ``.env``. The
backend's own processes see these variables because ``app.py`` loads ``.env``
into the environment at startup.

Controller (the AutoGPT backend that runs the copilot tools):

- ``COPILOT_LINK_PRIVATE_CHECKOUT=true`` turns the checkout tools on, with the
  broker running in-process (self-hosted, single host).
- ``COPILOT_LINK_HOSTED_CHECKOUT=true`` is also required on a cloud
  deployment, which additionally needs a remote broker and the registered
  Link OAuth client.
- ``COPILOT_LINK_LIVE_PAYMENTS=true`` allows live (non-test) spend requests;
  the broker must allow them too, and a browser fills a live card only where
  ``CHECKOUT_HTTPS_PROXY`` restricts its egress (``live_payments_allowed``).
- ``CHECKOUT_BROKER_ROUTES_FILE`` (one broker per user) or ``CHECKOUT_BROKER_URL``
  with ``_CA``, ``_CLIENT_CERT``, ``_CLIENT_KEY`` and ``_SECRET_FILE`` select a
  remote broker; see ``broker_routing``.

Broker and worker: ``CHECKOUT_BROKER_LEDGER_DIR`` (durable, private checkout
ledger) and ``CHECKOUT_HTTPS_PROXY`` (the restricted egress proxy).
"""

import os


def _flag(name: str) -> bool:
    return os.environ.get(name, "").lower() == "true"


def private_checkout_requested() -> bool:
    return _flag("COPILOT_LINK_PRIVATE_CHECKOUT")


def hosted_checkout_requested() -> bool:
    return _flag("COPILOT_LINK_HOSTED_CHECKOUT")


def live_payments_enabled() -> bool:
    return _flag("COPILOT_LINK_LIVE_PAYMENTS")


def https_proxy() -> str | None:
    return os.environ.get("CHECKOUT_HTTPS_PROXY") or None


def live_payments_allowed() -> bool:
    """Whether this browser host may fill a live card. Without the egress
    proxy, a checkout page could send the card anywhere; test cards cannot be
    charged, so test mode needs neither."""
    return live_payments_enabled() and https_proxy() is not None


def ledger_dir() -> str | None:
    return os.environ.get("CHECKOUT_BROKER_LEDGER_DIR") or None

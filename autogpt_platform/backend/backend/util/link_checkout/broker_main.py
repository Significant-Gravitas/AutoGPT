"""Entry point of a hosted checkout broker (``python -m ...broker_main``).

Runs from the standard backend image; see docs/platform/link-private-checkout.md
for the container settings and certificates it expects.
"""

import os
import ssl

import uvicorn

from backend.util.link_checkout.broker_service import configured_app
from backend.util.link_checkout.config import https_proxy, ledger_dir
from backend.util.link_checkout.runtime import require_runtime


def main() -> None:
    if not ledger_dir() or not https_proxy():
        raise RuntimeError(
            "Broker requires a durable ledger and restricted egress proxy"
        )
    require_runtime()
    uvicorn.run(
        configured_app(),
        host="0.0.0.0",
        port=8443,
        ssl_keyfile=os.environ["CHECKOUT_BROKER_SERVER_KEY"],
        ssl_certfile=os.environ["CHECKOUT_BROKER_SERVER_CERT"],
        ssl_ca_certs=os.environ["CHECKOUT_BROKER_CLIENT_CA"],
        ssl_cert_reqs=ssl.CERT_REQUIRED,
        access_log=False,
        log_config=None,
        limit_concurrency=20,
        timeout_keep_alive=5,
        h11_max_incomplete_event_size=32768,
    )


if __name__ == "__main__":
    main()

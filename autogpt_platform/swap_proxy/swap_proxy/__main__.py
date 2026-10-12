"""Run the proxy: mitmproxy in SOCKS5 mode with the swap addon."""

import asyncio
import logging
import os

from mitmproxy.options import Options
from mitmproxy.tools.dump import DumpMaster
from redis.asyncio.cluster import ClusterNode, RedisCluster

from swap_proxy.addon import MAX_BODY_BYTES, SwapProxyAddon
from swap_proxy.egress import EgressGuard
from swap_proxy.owners import OwnerDirectory
from swap_proxy.settings import Settings
from swap_proxy.source import BackendCredentialSource

logger = logging.getLogger(__name__)


def build_master(
    addon: SwapProxyAddon, listen_host: str, listen_port: int, confdir: str
):
    """A mitmproxy master that serves SOCKS5 and authenticates through *addon*."""
    options = Options(
        mode=[f"socks5@{listen_host}:{listen_port}"],
        confdir=os.path.expanduser(confdir),
        # The swap trusts the upstream certificate check; never turn it off.
        ssl_insecure=False,
    )
    master = DumpMaster(options, with_termlog=True, with_dumper=False)
    # mitmproxy's SOCKS5 layer asks the client for a username and password
    # only when ``proxyauth`` is set, and its built-in addon of that name would
    # then also demand credentials of its own on every request.  Remove the
    # addon and keep the option: the value is never compared to anything, the
    # pair is judged by SwapProxyAddon.socks5_auth alone.
    master.addons.remove(master.addons.get("proxyauth"))
    master.addons.add(addon)
    master.options.update(
        proxyauth="per-box",
        # A proxy carrying every box's downloads cannot hold them in memory.
        # The addon decides what that means for a body that needed a swap or a
        # scrub; see "Bodies and streaming" in ``addon.py``.
        stream_large_bodies=str(MAX_BODY_BYTES),
        # Upstream first, so that by the time a request is read the name its
        # certificate was verified for is known.
        connection_strategy="eager",
    )
    return master


CA_FILE = "mitmproxy-ca.pem"
DHPARAM_FILE = "mitmproxy-dhparam.pem"


class MissingCA(RuntimeError):
    pass


def require_ca(confdir: str, generate: bool) -> None:
    """Refuse to start without the provisioned CA.

    mitmproxy generates one when the directory has none.  The boxes trust one
    certificate, so a CA made up by a restarted or second replica breaks TLS
    to every bound host inside every box, and reads there as a certificate
    bug, not as missing provisioning.  Better not to come up at all.
    """
    directory = os.path.expanduser(confdir)
    path = os.path.join(directory, CA_FILE)
    if os.path.isfile(path):
        dhparam = os.path.join(directory, DHPARAM_FILE)
        if not os.path.isfile(dhparam) and not os.access(directory, os.W_OK):
            # mitmproxy writes this file next to the CA when it is missing.
            raise MissingCA(
                f"{directory} is read-only and has no {DHPARAM_FILE}. Mount it "
                f"beside {CA_FILE} (it is not a secret: fixed public parameters "
                "mitmproxy writes out with every CA), or make the directory writable."
            )
        return
    if generate:
        logger.warning("No CA at %s: generating one. Local use only.", path)
        return
    raise MissingCA(
        f"No signing CA at {path}. Mount the provisioned CA (private key and "
        f"certificate in one PEM file named {CA_FILE}) into SWAP_PROXY_CONFDIR; "
        "every replica must get the same one. For a local run, set "
        "SWAP_PROXY_GENERATE_CA=true."
    )


def connect_redis(settings: Settings) -> RedisCluster:
    def remap(address: tuple[str, int]) -> tuple[str, int]:
        # As the backend does: pin every shard to the seed host unless the
        # announced names resolve from here.
        if settings.redis_use_announced_address:
            return address
        return settings.redis_host, address[1]

    return RedisCluster(
        startup_nodes=[ClusterNode(settings.redis_host, settings.redis_port)],
        password=settings.redis_password,
        socket_timeout=5,
        socket_connect_timeout=5,
        address_remap=remap,
    )


async def run() -> None:
    settings = Settings()
    require_ca(settings.confdir, settings.generate_ca)
    redis = connect_redis(settings)
    await redis.ping()
    source = BackendCredentialSource(settings.backend_url)
    addon = SwapProxyAddon(
        OwnerDirectory(redis), source, EgressGuard(settings.egress_allow)
    )
    master = build_master(
        addon, settings.listen_host, settings.listen_port, settings.confdir
    )
    logger.info("Swap proxy on %s:%s", settings.listen_host, settings.listen_port)
    try:
        await master.run()
    finally:
        await source.aclose()
        await redis.aclose()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("SWAP_PROXY_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()

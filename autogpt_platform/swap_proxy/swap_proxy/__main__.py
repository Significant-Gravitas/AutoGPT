"""Run the proxy: mitmproxy in SOCKS5 mode with the swap addon."""

import asyncio
import logging
import os

from mitmproxy.options import Options
from mitmproxy.tools.dump import DumpMaster
from redis.asyncio.cluster import ClusterNode, RedisCluster

from swap_proxy.addon import SwapProxyAddon
from swap_proxy.egress import EgressGuard
from swap_proxy.owners import OwnerDirectory
from swap_proxy.settings import Settings
from swap_proxy.source import BackendCredentialSource

logger = logging.getLogger(__name__)

# Bodies over this are streamed through, neither swapped nor scrubbed: a
# proxy carrying every box's downloads cannot hold them in memory.
STREAM_BODIES_OVER = "5m"


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
        stream_large_bodies=STREAM_BODIES_OVER,
        # Upstream first, so that by the time a request is read the name its
        # certificate was verified for is known.
        connection_strategy="eager",
    )
    return master


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

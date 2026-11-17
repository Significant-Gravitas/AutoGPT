"""Which site each frame of the checkout tab belongs to, and where a payment
control may sit.

A site is a registrable domain ("example.co.uk" for "pay.example.co.uk"),
read from the Public Suffix List. The customer approves paying on the checkout
page's site, so a payment control may sit there or in a known card
processor's frame, inside no frame from any other site.
"""

import functools
import ipaddress
from urllib.parse import urlsplit

from publicsuffixlist import PublicSuffixList

from backend.util.link_checkout.cdp_models import FrameContext
from backend.util.link_checkout.refusals import FRAME_ELSEWHERE, CheckoutRefused

# Sites whose frames take card details on other sites' checkout pages. Any
# other site's frame on the page (an ad, a chat widget, a seller's embed) may
# not hold a payment control.
CARD_PROCESSOR_SITES = frozenset(
    {
        "adyen.com",
        "authorize.net",
        "braintreegateway.com",
        "checkout.com",
        "cybersource.com",
        "paypal.com",
        "recurly.com",
        "shopifyinc.com",
        "squarecdn.com",
        "squareup.com",
        "stripe.com",
    }
)


def check_frame_site(
    context: FrameContext, frames: list[FrameContext], checkout_url: str
) -> None:
    """A payment control must sit on the checkout page's site or in a known
    card processor's frame, and every frame around it on one of those two
    sites. So neither a third party's frame nor a processor frame that a third
    party embeds for its own account can take the card."""
    sites = frame_sites(context, frames)
    if sites is None:
        raise CheckoutRefused(FRAME_ELSEWHERE)
    page, own = site(checkout_url), sites[-1]
    if own != page and own not in CARD_PROCESSOR_SITES:
        raise CheckoutRefused(FRAME_ELSEWHERE)
    if any(around not in {page, own} for around in sites[:-1]):
        raise CheckoutRefused(FRAME_ELSEWHERE)


def frame_sites(context: FrameContext, frames: list[FrameContext]) -> list[str] | None:
    """The sites of a frame and of every frame around it, the tab's top frame
    first; None when its chain of frames doesn't reach the top. An about:blank
    or about:srcdoc frame takes the origin of the frame holding it, so it takes
    that frame's site too."""
    by_id = {f.frame.id: f.frame for f in frames}
    chain = [context.frame]
    while chain[-1].parentId:
        parent = by_id.get(chain[-1].parentId)
        if parent is None or len(chain) > len(by_id):
            return None
        chain.append(parent)
    sites: list[str] = []
    for frame in reversed(chain):
        inherits = bool(sites) and frame.url.startswith("about:")
        sites.append(sites[-1] if inherits else site(frame.url))
    return sites


def site(url: str) -> str:
    """The registrable domain of an address; an IP address, or a host with
    none, stands alone."""
    host = (urlsplit(url).hostname or "").rstrip(".")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        return _public_suffixes().privatesuffix(host) or host


# The list ships inside the package; parsed on first use.
@functools.cache
def _public_suffixes() -> PublicSuffixList:
    return PublicSuffixList()

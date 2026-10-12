const CHECKOUT_CANCELLED_PARAM = "paywall_checkout";
const CHECKOUT_CANCELLED_VALUE = "cancelled";

// Backing out of Stripe must land on the page the user was on, so the gate
// re-applies. The marker is the only way that return is told apart from any
// other visit to the same page. It only feeds analytics, so an unparseable
// URL falls back to the plain page rather than blocking checkout.
export function buildPaywallCancelUrl(href: string) {
  try {
    const url = new URL(href);
    url.searchParams.set(CHECKOUT_CANCELLED_PARAM, CHECKOUT_CANCELLED_VALUE);
    return url.toString();
  } catch {
    return href;
  }
}

// Reads the marker once and drops it from the address bar, so a refresh or a
// shared link does not report the same abandonment again. The state is null
// on purpose: Next's own history state would make it skip syncing its router
// URL, and its next update would write the marker back.
export function consumePaywallCheckoutCancel() {
  try {
    const url = new URL(window.location.href);
    if (
      url.searchParams.get(CHECKOUT_CANCELLED_PARAM) !==
      CHECKOUT_CANCELLED_VALUE
    )
      return false;
    url.searchParams.delete(CHECKOUT_CANCELLED_PARAM);
    window.history.replaceState(
      null,
      "",
      `${url.pathname}${url.search}${url.hash}`,
    );
    return true;
  } catch {
    return false;
  }
}

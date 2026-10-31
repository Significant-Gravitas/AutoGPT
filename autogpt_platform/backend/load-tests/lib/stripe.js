// A paid subscription for a throwaway account, in Stripe test mode, so a
// target that paywalls new accounts lets the run's accounts through. The
// hosted Checkout page cannot be driven from k6, so setup() does what the
// page would: the backend creates the Stripe customer and a Checkout session
// for the cheapest tier it prices monthly, and the Stripe test API attaches
// a test card to that customer and starts a subscription on the session's
// price. One request through the backend's lazy Stripe reconcile then copies
// the tier onto the user, and the paywall opens for that account.
//
// STRIPE_TEST_SECRET_KEY is a test-mode key (sk_test_ or rk_test_) of the
// Stripe account the target uses. It is read from the environment, checked
// by prefix only, sent to api.stripe.com and nowhere else, and never logged
// or written to a summary. Without it, nothing here runs.
//
// teardown() cancels the subscriptions the run created, and so does a
// setup() that throws (lib/seed.js). A run stopped before its teardown, or
// a setup() that runs out of time, leaves them behind, each carrying the run
// id in its metadata; scripts/cancel-load-test-subscriptions.js cancels
// those.
import http from "k6/http";
import { Counter } from "k6/metrics";

const STRIPE_API = "https://api.stripe.com/v1";
const TEST_CARD = "pm_card_visa";
export const RUN_METADATA_KEY = "autogpt_load_test_run";

export const subscriptionsCreated = new Counter("seed_subscriptions");
export const subscriptionsCancelled = new Counter(
  "seed_subscriptions_cancelled",
);

// The key, or null when unset. Anything but a test-mode key is refused, and
// so is a target with any URL at production (configs/environment.js); the
// cleanup script, which has no target, passes null.
export function stripeTestKey(target) {
  const key = __ENV.STRIPE_TEST_SECRET_KEY || "";
  if (key === "") return null;
  if (!/^(sk|rk)_test_/.test(key)) {
    throw new Error(
      "STRIPE_TEST_SECRET_KEY is not a Stripe test-mode key (sk_test_ or rk_test_); refusing to use it",
    );
  }
  if (target && target.PRODUCTION) {
    throw new Error(
      "STRIPE_TEST_SECRET_KEY is not used against production; unset it",
    );
  }
  return key;
}

// Subscribes one account at the target: returns { ok, tier, subscriptionId,
// customerId, reason }. subscriptionId is set as soon as a subscription
// exists, whether or not the target picked it up, so teardown can cancel it.
export function subscribeAccount(target, headers, runId) {
  const api = target.API_BASE_URL;
  const status = backend(
    "GET",
    `${api}/credits/subscription`,
    headers,
    null,
    "/api/credits/subscription",
  );
  if (!status.ok || !status.json) {
    return failure(
      `reading the subscription status failed (${describe(status)})`,
    );
  }
  if (status.json.tier && status.json.tier !== "NO_TIER") {
    return { ok: true, tier: status.json.tier, already: true };
  }
  const tier = cheapestTier(status.json.tier_costs);
  if (!tier) {
    return failure(
      "the target prices no subscription tier monthly (tier_costs has no positive amount)",
    );
  }

  // The backend creates the customer and the Checkout session for the tier;
  // the session's id is in the URL it answers with.
  const checkout = backend(
    "POST",
    `${api}/credits/subscription`,
    headers,
    {
      tier,
      billing_cycle: "monthly",
      success_url: `${target.FRONTEND_URL}/profile/credits`,
      cancel_url: `${target.FRONTEND_URL}/profile/credits`,
    },
    "/api/credits/subscription",
  );
  if (!checkout.ok || !checkout.json) {
    return failure(
      `creating the checkout session for ${tier} failed (${describe(checkout)})`,
    );
  }
  const match = /cs_(test|live)_[A-Za-z0-9]+/.exec(
    String(checkout.json.url || ""),
  );
  if (!match) return failure("the checkout answer carried no session id");
  const sessionId = match[0];

  const session = stripe(
    "GET",
    `/checkout/sessions/${sessionId}?${encodeURIComponent("expand[]")}=line_items`,
    null,
    "checkout_session",
  );
  if (!session.ok || !session.json) {
    return failure(
      `retrieving the checkout session failed (${stripeError(session)}); is STRIPE_TEST_SECRET_KEY for the Stripe account the target uses?`,
    );
  }
  if (session.json.livemode === true) {
    return failure("the checkout session is live mode; refusing to pay it");
  }
  const customerId = session.json.customer;
  const items = (session.json.line_items && session.json.line_items.data) || [];
  const priceId = items.length > 0 && items[0].price ? items[0].price.id : null;
  if (!customerId || !priceId) {
    return failure("the checkout session names no customer or no price");
  }

  const attached = stripe(
    "POST",
    `/payment_methods/${TEST_CARD}/attach`,
    { customer: customerId },
    "payment_method_attach",
  );
  if (!attached.ok || !attached.json) {
    return failure(`attaching the test card failed (${stripeError(attached)})`);
  }
  // Attaching the test card alias creates a real payment method; its id is
  // what a subscription can use.
  let paymentMethodId = attached.json.id;
  if (!paymentMethodId || paymentMethodId.indexOf("pm_card_") === 0) {
    const listed = stripe(
      "GET",
      `/payment_methods?customer=${customerId}&type=card&limit=1`,
      null,
      "payment_method_list",
    );
    paymentMethodId =
      listed.ok && listed.json && listed.json.data && listed.json.data[0]
        ? listed.json.data[0].id
        : null;
  }
  if (!paymentMethodId) {
    return failure(
      "no payment method on the customer after attaching the test card",
    );
  }

  const params = {
    customer: customerId,
    "items[0][price]": priceId,
    default_payment_method: paymentMethodId,
  };
  params[`metadata[${RUN_METADATA_KEY}]`] = runId;
  const created = stripe(
    "POST",
    "/subscriptions",
    params,
    "subscription_create",
  );
  if (!created.ok || !created.json || !created.json.id) {
    return failure(
      `creating the subscription failed (${stripeError(created)})`,
    );
  }
  const subscriptionId = created.json.id;
  subscriptionsCreated.add(1);
  if (created.json.status !== "active") {
    return failure(
      `the subscription is ${created.json.status}, not active`,
      subscriptionId,
      customerId,
    );
  }

  // The backend copies the tier from Stripe on the first lookup that reaches
  // its reconcile; the copilot usage read is one.
  backend("GET", `${api}/chat/usage`, headers, null, "/api/chat/usage");
  const after = backend(
    "GET",
    `${api}/credits/subscription`,
    headers,
    null,
    "/api/credits/subscription",
  );
  const tierNow = after.ok && after.json ? after.json.tier : null;
  if (tierNow !== tier) {
    return failure(
      `the subscription is active in Stripe but the target reports tier ${tierNow}`,
      subscriptionId,
      customerId,
    );
  }
  return { ok: true, tier, subscriptionId, customerId };
}

// Cancels subscriptions at once; returns how many were and were not.
export function cancelSubscriptions(ids) {
  const result = { cancelled: 0, failed: 0 };
  for (const id of ids) {
    const res = stripe(
      "DELETE",
      `/subscriptions/${id}`,
      null,
      "subscription_cancel",
      "teardown",
    );
    if (res.ok) {
      result.cancelled++;
      subscriptionsCancelled.add(1);
    } else {
      result.failed++;
      console.warn(
        `cancelling subscription ${id} failed (${stripeError(res)})`,
      );
    }
  }
  return result;
}

// Active subscriptions that load-test runs created, found by their metadata:
// every run's, or one run's when runId is given.
export function listRunSubscriptions(runId) {
  const found = [];
  let startingAfter = null;
  for (let page = 0; page < 100; page++) {
    const query = `status=active&limit=100${startingAfter ? `&starting_after=${startingAfter}` : ""}`;
    const res = stripe(
      "GET",
      `/subscriptions?${query}`,
      null,
      "subscription_list",
      "teardown",
    );
    if (!res.ok || !res.json || !Array.isArray(res.json.data)) {
      throw new Error(`listing subscriptions failed (${stripeError(res)})`);
    }
    for (const sub of res.json.data) {
      const tag = sub.metadata && sub.metadata[RUN_METADATA_KEY];
      if (tag && (!runId || tag === runId))
        found.push({ id: sub.id, run: tag });
    }
    if (!res.json.has_more || res.json.data.length === 0) break;
    startingAfter = res.json.data[res.json.data.length - 1].id;
  }
  return found;
}

// The cheapest tier with a monthly price, which is what the checkout above
// asks for. tier_costs holds monthly amounts and 0 for a tier priced only
// yearly, which a monthly checkout cannot buy.
function cheapestTier(tierCosts) {
  if (!tierCosts || typeof tierCosts !== "object") return null;
  let best = null;
  for (const name of Object.keys(tierCosts)) {
    const cost = Number(tierCosts[name]);
    if (!Number.isFinite(cost) || cost <= 0) continue;
    if (best === null || cost < best.cost) best = { name, cost };
  }
  return best ? best.name : null;
}

function failure(reason, subscriptionId, customerId) {
  return {
    ok: false,
    reason,
    subscriptionId: subscriptionId || null,
    customerId: customerId || null,
  };
}

function stripe(method, path, params, label, phase) {
  const res = http.request(method, `${STRIPE_API}${path}`, params || null, {
    headers: { Authorization: `Bearer ${__ENV.STRIPE_TEST_SECRET_KEY}` },
    tags: { name: `stripe:${label}`, phase: phase || "setup" },
    responseCallback: http.expectedStatuses(200),
    timeout: "60s",
  });
  return { res, status: res.status, ok: res.status === 200, json: parse(res) };
}

function backend(method, url, headers, body, name) {
  const res = http.request(method, url, body ? JSON.stringify(body) : null, {
    headers,
    tags: { name, phase: "setup" },
    responseCallback: http.expectedStatuses(200),
    timeout: "60s",
  });
  return { res, status: res.status, ok: res.status === 200, json: parse(res) };
}

function parse(res) {
  try {
    return res.json();
  } catch (e) {
    return null;
  }
}

function describe(result) {
  if (result.status === 0) return result.res.error || "no response";
  const body = result.json;
  const text = (body && (body.detail || body.message)) || "";
  return `HTTP ${result.status}${text ? ` ${String(text).slice(0, 160)}` : ""}`;
}

// Stripe's error text can quote part of the key it rejected; nothing that
// looks like a key leaves this module.
function stripeError(result) {
  if (result.status === 0) return scrub(result.res.error || "no response");
  const err = result.json && result.json.error;
  const text = err
    ? [err.type, err.code, err.message].filter(Boolean).join(" ")
    : "";
  return `Stripe HTTP ${result.status}${text ? ` ${scrub(text).slice(0, 160)}` : ""}`;
}

function scrub(text) {
  return String(text).replace(/[sr]k_(test|live)_[A-Za-z0-9_*]*/g, "[key]");
}

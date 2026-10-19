import {
  fetchVapidPublicKey,
  removeSubscriptionFromServer,
  sendSubscriptionToServer,
} from "./api";
import {
  isPushSupported,
  registerServiceWorker,
  subscribeToPush,
  unsubscribeFromPush,
} from "./registration";

export async function setupPushSubscription(): Promise<boolean> {
  if (!isPushSupported()) return false;
  if (Notification.permission !== "granted") return false;

  const registration = await registerServiceWorker();
  if (!registration) return false;

  await navigator.serviceWorker.ready;

  const vapidKey =
    process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY || (await fetchVapidPublicKey());
  if (!vapidKey) return false;

  const subscription = await subscribeToPush(registration, vapidKey);
  if (!subscription) return false;

  return sendSubscriptionToServer(subscription);
}

export async function teardownPushSubscription(): Promise<void> {
  if (!navigator.serviceWorker) return;
  const registration = await navigator.serviceWorker.getRegistration("/");
  if (!registration) return;
  const subscription = await registration.pushManager.getSubscription();
  if (!subscription) return;

  // Both at once, on the subscription read above. The browser side is local
  // and fast, so a slow server can't leave this browser subscribed, and a
  // late finish can never touch a subscription the next account has made.
  // A row the server misses points at a dead endpoint and is dropped on the
  // next send.
  await Promise.all([
    unsubscribeFromPush(subscription),
    removeSubscriptionFromServer(subscription.endpoint),
  ]);
}

const LOGOUT_TEARDOWN_TIMEOUT_MS = 3_000;

/**
 * Remove this browser's push subscription while the session is still valid.
 * The `/logout` page sits outside the platform layout, so the provider's own
 * teardown never sees the user turn null; without this the next account to
 * sign in here keeps receiving the previous account's notifications. Bounded
 * so a slow or offline network never blocks the logout itself.
 */
export async function teardownPushSubscriptionBeforeLogout(): Promise<void> {
  await Promise.race([
    teardownPushSubscription().catch((error) =>
      console.error("Push notification teardown failed:", error),
    ),
    new Promise<void>((resolve) =>
      setTimeout(resolve, LOGOUT_TEARDOWN_TIMEOUT_MS),
    ),
  ]);
}

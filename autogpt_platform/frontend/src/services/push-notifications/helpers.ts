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
import { withPushSubscriptionLock } from "./subscription-lock";

const LOGOUT_TEARDOWN_TIMEOUT_MS = 3_000;
const TEARDOWN_LOCK_WAIT_MS = 2_000;

/**
 * `isCancelled` lets the caller drop a setup the user has since overtaken, by
 * switching notifications off or signing out. It is checked inside the lock,
 * so a teardown that got the lock first is never undone by a late subscribe.
 */
export async function setupPushSubscription(
  isCancelled: () => boolean = () => false,
): Promise<boolean> {
  if (!isPushSupported()) return false;
  if (Notification.permission !== "granted") return false;

  const registration = await registerServiceWorker();
  if (!registration) return false;

  await navigator.serviceWorker.ready;

  const vapidKey =
    process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY || (await fetchVapidPublicKey());
  if (!vapidKey) return false;

  const subscription = await withPushSubscriptionLock(async () => {
    if (isCancelled()) return null;
    const existing = await registration.pushManager
      .getSubscription()
      .catch(() => null);
    const result = await subscribeToPush(registration, vapidKey);
    if (!result || !isCancelled()) return result;
    // A teardown that gave up waiting for the lock may already have run and
    // found nothing, so a subscription this call made after the cancel is
    // ours to drop. One that was already there is shared with the other
    // tabs, and a cancel can also mean this tab just unmounted.
    if (result.endpoint !== existing?.endpoint) {
      await unsubscribeFromPush(result);
    }
    return null;
  });
  if (!subscription || isCancelled()) return false;

  return sendSubscriptionToServer(subscription);
}

export async function teardownPushSubscription(): Promise<void> {
  if (!navigator.serviceWorker) return;
  const registration = await navigator.serviceWorker.getRegistration("/");
  if (!registration) return;
  // The server call starts at once but runs outside the lock: only the
  // browser side has to finish before another tab may subscribe. Unsubscribe
  // the subscription read here, never a later one, so a slow server can't
  // leave this browser subscribed or touch the next account's subscription.
  // A row the server misses points at a dead endpoint and is dropped on the
  // next send.
  let serverRemoval: Promise<boolean> | undefined;
  await withPushSubscriptionLock(
    async () => {
      const subscription = await registration.pushManager.getSubscription();
      if (!subscription) return;
      serverRemoval = removeSubscriptionFromServer(subscription.endpoint);
      await unsubscribeFromPush(subscription);
    },
    // Inside the logout bound, so a tab stuck holding the lock can't make
    // logout finish before this browser is unsubscribed.
    { maxWaitMs: TEARDOWN_LOCK_WAIT_MS },
  );
  await serverRemoval;
}

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

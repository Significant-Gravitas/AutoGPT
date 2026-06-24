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

  // Tell the backend first — if the row is deleted but the browser keeps the
  // subscription, future pushes 404 and get cleaned up anyway.
  await removeSubscriptionFromServer(subscription.endpoint);
  await unsubscribeFromPush(registration);
}

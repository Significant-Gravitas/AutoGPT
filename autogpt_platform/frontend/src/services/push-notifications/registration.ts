const SW_PATH = "/push-sw.js";

/**
 * Convert a base64url VAPID key to a Uint8Array for the Push API.
 */
function urlBase64ToUint8Array(base64String: string): ArrayBuffer {
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const rawData = window.atob(base64);
  const arr = new Uint8Array(rawData.length);
  for (let i = 0; i < rawData.length; i++) {
    arr[i] = rawData.charCodeAt(i);
  }
  return arr.buffer as ArrayBuffer;
}

export function isPushSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    "serviceWorker" in navigator &&
    "PushManager" in window &&
    "Notification" in window
  );
}

export async function registerServiceWorker(): Promise<ServiceWorkerRegistration | null> {
  if (!("serviceWorker" in navigator)) return null;
  try {
    return await navigator.serviceWorker.register(SW_PATH, { scope: "/" });
  } catch (error) {
    console.error("Failed to register push service worker:", error);
    return null;
  }
}

export async function subscribeToPush(
  registration: ServiceWorkerRegistration,
  vapidPublicKey: string,
): Promise<PushSubscription | null> {
  try {
    const existing = await registration.pushManager.getSubscription();
    if (existing) return existing;

    return await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(vapidPublicKey),
    });
  } catch (error) {
    console.error("Failed to subscribe to push:", error);
    return null;
  }
}

export async function unsubscribeFromPush(
  registration: ServiceWorkerRegistration,
): Promise<boolean> {
  try {
    const subscription = await registration.pushManager.getSubscription();
    if (subscription) {
      return await subscription.unsubscribe();
    }
    return true;
  } catch (error) {
    console.error("Failed to unsubscribe from push:", error);
    return false;
  }
}

const PROXY_BASE = "/api/proxy/api/push";

export async function fetchVapidPublicKey(): Promise<string | null> {
  try {
    const response = await fetch(`${PROXY_BASE}/vapid-key`);
    if (!response.ok) return null;
    const data = await response.json();
    return data.public_key || null;
  } catch {
    return null;
  }
}

export async function sendSubscriptionToServer(
  subscription: PushSubscription,
): Promise<boolean> {
  const json = subscription.toJSON();
  try {
    const response = await fetch(`${PROXY_BASE}/subscribe`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        endpoint: json.endpoint,
        keys: {
          p256dh: json.keys?.p256dh ?? "",
          auth: json.keys?.auth ?? "",
        },
        user_agent: navigator.userAgent,
      }),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export async function removeSubscriptionFromServer(
  endpoint: string,
): Promise<boolean> {
  try {
    const response = await fetch(`${PROXY_BASE}/unsubscribe`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ endpoint }),
    });
    return response.ok;
  } catch {
    return false;
  }
}

const PUSH_SUBSCRIPTION_LOCK = "autogpt-push-subscription";

/**
 * Run `action` holding a lock shared by every tab of this browser. All tabs
 * share one push subscription, so a tab subscribing for the next account
 * must not read it while another tab is still unsubscribing it for the last.
 *
 * Every browser with the Push API also has Web Locks (Safari added locks in
 * 15.4, push in 16), so the unlocked fallback only runs where push does not.
 */
export async function withPushSubscriptionLock<T>(
  action: () => Promise<T>,
): Promise<T> {
  if (typeof navigator === "undefined" || !navigator.locks) return action();
  let result!: T;
  await navigator.locks.request(PUSH_SUBSCRIPTION_LOCK, async () => {
    result = await action();
  });
  return result;
}

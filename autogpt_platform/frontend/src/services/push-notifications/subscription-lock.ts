const PUSH_SUBSCRIPTION_LOCK = "autogpt-push-subscription";

interface Options {
  /**
   * Stop waiting for the lock after this long and run `action` unlocked.
   * For logout, where removing the subscription late or racily still beats
   * not removing it before the tab closes.
   */
  maxWaitMs?: number;
}

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
  { maxWaitMs }: Options = {},
): Promise<T> {
  if (typeof navigator === "undefined" || !navigator.locks) return action();

  const controller = new AbortController();
  const timer =
    maxWaitMs === undefined
      ? undefined
      : setTimeout(() => controller.abort(), maxWaitMs);
  let granted = false;
  let result!: T;
  try {
    await navigator.locks.request(
      PUSH_SUBSCRIPTION_LOCK,
      { signal: controller.signal },
      async () => {
        granted = true;
        clearTimeout(timer);
        result = await action();
      },
    );
  } catch (error) {
    if (granted || !controller.signal.aborted) throw error;
    return action();
  } finally {
    clearTimeout(timer);
  }
  return result;
}

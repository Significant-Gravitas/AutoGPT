/** Rejects once `ms` pass, so a stalled call ends instead of waiting for ever. */
export function withDeadline<T>(promise: Promise<T>, ms: number, onTimeout?: () => void) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      onTimeout?.();
      reject(new Error("Timed out"));
    }, ms);
  });
  return Promise.race([promise, deadline]).finally(() => clearTimeout(timer));
}

/** A request given `ms` to answer; past that its fetch is aborted, not left open. */
export function requestWithin<T>(ms: number, request: (signal: AbortSignal) => Promise<T>) {
  const controller = new AbortController();
  return withDeadline(request(controller.signal), ms, () => controller.abort());
}

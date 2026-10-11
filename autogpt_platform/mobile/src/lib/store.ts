/**
 * A tiny observable value for `useSyncExternalStore`. Each `set` publishes a
 * new snapshot object, so React re-renders only when state actually changed.
 */
export class Observable<T> {
  private listeners = new Set<() => void>();

  constructor(private value: T) {}

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = (): T => this.value;

  protected set(patch: Partial<T>) {
    this.value = { ...this.value, ...patch };
    for (const listener of this.listeners) listener();
  }
}

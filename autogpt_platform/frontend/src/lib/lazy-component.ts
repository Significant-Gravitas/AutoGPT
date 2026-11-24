import { type ComponentType, createElement, useSyncExternalStore } from "react";

/**
 * A component whose module loads on first use, or earlier through `preload`.
 * Until `useIsLoaded` is true the caller draws a stand-in, which, unlike
 * `next/dynamic`'s fallback, can use the component's props; then `Loaded`.
 */
export function lazyComponent<Props extends object>(
  load: () => Promise<ComponentType<Props>>,
) {
  let loaded: ComponentType<Props> | null = null;
  let loading: Promise<void> | null = null;
  const listeners = new Set<() => void>();

  function preload() {
    loading ??= load().then(
      (component) => {
        loaded = component;
        listeners.forEach((notify) => notify());
      },
      () => {
        // A failed chunk load is retried by the next mount.
        loading = null;
      },
    );
    return loading;
  }

  function subscribe(notify: () => void) {
    listeners.add(notify);
    void preload();
    return () => {
      listeners.delete(notify);
    };
  }

  function isLoaded() {
    return loaded !== null;
  }

  function useIsLoaded() {
    return useSyncExternalStore(subscribe, isLoaded, isNeverLoadedOnServer);
  }

  function Loaded(props: Props) {
    if (!loaded) throw new Error("Rendered before useIsLoaded() was true");
    return createElement(loaded, props);
  }

  return { preload, useIsLoaded, Loaded };
}

function isNeverLoadedOnServer() {
  return false;
}

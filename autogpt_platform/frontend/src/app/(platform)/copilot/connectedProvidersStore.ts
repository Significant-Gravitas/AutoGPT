import { create } from "zustand";

/**
 * Session-scoped record of credential providers the user has connected
 * during the current chat. Lets credential cards self-dismiss when an
 * earlier card in the same session has already satisfied the requirement,
 * killing the "duplicate sign-in card" UX problem and obsoleting the local
 * `isDismissed` workarounds that callers used to add to survive remounts.
 *
 * Entries are keyed by `${sessionID}::${provider}` so multiple chats can
 * coexist without cross-contamination, and so a different chat reading the
 * same provider name does not falsely auto-dismiss.
 */
interface ConnectedProvidersState {
  connected: Set<string>;
  /**
   * Tracks `(sessionID, sortedProviders)` combinations that have already
   * triggered an auto-dismiss send. Prevents N parallel cards for the same
   * provider-set from each firing an identical "Please proceed" message
   * when one card connects the provider and the others re-render.
   */
  autoDismissedKeys: Set<string>;
  markConnected(args: { sessionID: string; providers: string[] }): void;
  /**
   * Atomically claim the auto-dismiss slot for `(sessionID, providers)`.
   * Returns `true` on the first call (caller should send the proceed
   * message) and `false` on subsequent calls (caller should silently
   * dismiss the card without sending).
   */
  tryClaimAutoDismiss(args: {
    sessionID: string;
    providers: string[];
  }): boolean;
  clearSession(sessionID: string): void;
}

function makeKey(sessionID: string, provider: string) {
  return `${sessionID}::${provider}`;
}

function makeAutoDismissKey(sessionID: string, providers: string[]) {
  const sorted = [...providers].sort().join(",");
  return `${sessionID}::${sorted}`;
}

export const useConnectedProvidersStore = create<ConnectedProvidersState>(
  (set) => ({
    connected: new Set<string>(),
    autoDismissedKeys: new Set<string>(),
    markConnected({ sessionID, providers }) {
      if (!sessionID || providers.length === 0) return;
      set((state) => {
        const next = new Set(state.connected);
        for (const p of providers) {
          if (p) next.add(makeKey(sessionID, p));
        }
        return { connected: next };
      });
    },
    tryClaimAutoDismiss({ sessionID, providers }) {
      if (!sessionID || providers.length === 0) return false;
      const key = makeAutoDismissKey(sessionID, providers);
      let claimed = false;
      set((state) => {
        if (state.autoDismissedKeys.has(key)) return state;
        claimed = true;
        const next = new Set(state.autoDismissedKeys);
        next.add(key);
        return { autoDismissedKeys: next };
      });
      return claimed;
    },
    clearSession(sessionID) {
      if (!sessionID) return;
      set((state) => {
        const prefix = `${sessionID}::`;
        const nextConnected = new Set<string>();
        for (const k of state.connected) {
          if (!k.startsWith(prefix)) nextConnected.add(k);
        }
        const nextAutoDismissed = new Set<string>();
        for (const k of state.autoDismissedKeys) {
          if (!k.startsWith(prefix)) nextAutoDismissed.add(k);
        }
        return {
          connected: nextConnected,
          autoDismissedKeys: nextAutoDismissed,
        };
      });
    },
  }),
);

export function useAreAllConnected(
  sessionID: string | null | undefined,
  providers: string[],
) {
  return useConnectedProvidersStore((state) => {
    if (!sessionID || providers.length === 0) return false;
    return providers.every((p) => state.connected.has(makeKey(sessionID, p)));
  });
}

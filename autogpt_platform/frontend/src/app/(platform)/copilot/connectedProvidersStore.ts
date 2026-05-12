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
  markConnected(args: { sessionID: string; providers: string[] }): void;
  clearSession(sessionID: string): void;
}

function makeKey(sessionID: string, provider: string) {
  return `${sessionID}::${provider}`;
}

export const useConnectedProvidersStore = create<ConnectedProvidersState>(
  (set) => ({
    connected: new Set<string>(),
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
    clearSession(sessionID) {
      if (!sessionID) return;
      set((state) => {
        const next = new Set<string>();
        const prefix = `${sessionID}::`;
        for (const k of state.connected) {
          if (!k.startsWith(prefix)) next.add(k);
        }
        return { connected: next };
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

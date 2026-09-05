import { useCallback, useEffect, useRef, useState } from "react";

import { detectMCPAuthScheme, type MCPAuthScheme } from "@/lib/mcp-auth";

/**
 * State machine behind the Bearer/Basic selector.
 *
 * `storedScheme` is the scheme the server already has for this credential, so
 * reconnecting an existing Basic credential does not silently downgrade it to
 * Bearer. It is only applied while the user has neither chosen a scheme nor
 * typed a credential; `locked` records that either has happened.
 *
 * The returned callbacks are stable across renders and read `storedScheme`
 * through a ref, so a caller that memoizes them (`MCPToolDialog` holds `reset`
 * and `applyDiscoveredTools` in `useCallback`) cannot pin the first render's
 * stored scheme — which was the pre-connection default, defeating Basic
 * seeding for every credential discovered afterwards.
 */
export function useMCPAuthScheme(
  storedScheme: MCPAuthScheme,
  credential: string,
) {
  const [scheme, setScheme] = useState<MCPAuthScheme>(storedScheme);
  const [locked, setLocked] = useState(false);
  const storedSchemeRef = useRef(storedScheme);
  storedSchemeRef.current = storedScheme;

  useEffect(() => {
    if (!locked && !credential.trim()) setScheme(storedScheme);
  }, [locked, storedScheme, credential]);

  /** The user picked a scheme; stop syncing from the stored one. */
  const selectScheme = useCallback((next: MCPAuthScheme) => {
    setScheme(next);
    setLocked(true);
  }, []);

  /** A pasted value carried an explicit scheme; honour and lock it. */
  const detectSchemeFrom = useCallback((value: string) => {
    const detected = detectMCPAuthScheme(value);
    if (detected) {
      setScheme(detected);
      setLocked(true);
    }
  }, []);

  /** Back to following the stored scheme, e.g. after switching servers. */
  const resetScheme = useCallback(() => {
    setScheme(storedSchemeRef.current);
    setLocked(false);
  }, []);

  return { scheme, selectScheme, detectSchemeFrom, resetScheme };
}

import { useCallback, useEffect, useRef, useState } from "react";

import { detectMCPAuthScheme, type MCPAuthScheme } from "@/lib/mcp-auth";

/**
 * Follows `storedScheme` until the user picks a scheme or types a credential.
 * The callbacks are stable and read `storedScheme` through a ref so memoized
 * callers never pin a stale value.
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

  const selectScheme = useCallback((next: MCPAuthScheme) => {
    setScheme(next);
    setLocked(true);
  }, []);

  const detectSchemeFrom = useCallback((value: string) => {
    const detected = detectMCPAuthScheme(value);
    if (detected) {
      setScheme(detected);
      setLocked(true);
    }
  }, []);

  const resetScheme = useCallback(() => {
    setScheme(storedSchemeRef.current);
    setLocked(false);
  }, []);

  return { scheme, selectScheme, detectSchemeFrom, resetScheme };
}

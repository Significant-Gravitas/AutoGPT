import { useEffect, useState } from "react";

import { detectMCPAuthScheme, type MCPAuthScheme } from "@/lib/mcp-auth";

/**
 * State machine behind the Bearer/Basic selector.
 *
 * `storedScheme` is the scheme the server already has for this credential, so
 * reconnecting an existing Basic credential does not silently downgrade it to
 * Bearer. It is only applied while the user has neither chosen a scheme nor
 * typed a credential — `locked` (previously named `touched`, which stopped
 * being true once paste-detection started setting it too) records that.
 */
export function useMCPAuthScheme(
  storedScheme: MCPAuthScheme,
  credential: string,
) {
  const [scheme, setScheme] = useState<MCPAuthScheme>(storedScheme);
  const [locked, setLocked] = useState(false);

  useEffect(() => {
    if (!locked && !credential.trim()) setScheme(storedScheme);
  }, [locked, storedScheme, credential]);

  return {
    scheme,
    /** The user picked a scheme; stop syncing from the stored one. */
    selectScheme(next: MCPAuthScheme) {
      setScheme(next);
      setLocked(true);
    },
    /** A pasted value carried an explicit scheme; honour and lock it. */
    detectSchemeFrom(value: string) {
      const detected = detectMCPAuthScheme(value);
      if (detected) {
        setScheme(detected);
        setLocked(true);
      }
    },
    /** Back to following the stored scheme, e.g. after switching servers. */
    resetScheme() {
      setScheme(storedScheme);
      setLocked(false);
    },
  };
}

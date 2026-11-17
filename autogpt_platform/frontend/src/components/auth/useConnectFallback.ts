import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import {
  type ConnectingProvider,
  isConnectingProvider,
  takeConnectAttempt,
} from "@/lib/auth/connect-attempt";

/**
 * The provider to offer signing in with, without connecting: one whose
 * connecting sign-in came back refused (`connect=` from the with-provider
 * route), or one this tab started and never finished.
 */
export function useConnectFallback() {
  const refused = useSearchParams().get("connect");
  const [provider, setProvider] = useState<ConnectingProvider | null>(null);

  // Session storage exists only in the browser, so it's read after mount.
  useEffect(() => {
    const abandoned = takeConnectAttempt();
    setProvider(isConnectingProvider(refused) ? refused : abandoned);
  }, [refused]);

  return provider;
}

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import {
  type ConnectingProvider,
  isConnectingProvider,
  takeConnectAttempt,
} from "@/lib/auth/connect-attempt";
import { usePageRestored } from "./usePageRestored";

// Reading the note removes it, so a second read finds nothing and keeps the
// first one's answer: React runs effects twice in development.
function takeAbandoned(setAbandoned: (provider: ConnectingProvider) => void) {
  const attempt = takeConnectAttempt();
  if (attempt) setAbandoned(attempt);
}

/**
 * The provider to offer signing in with, without connecting: one whose
 * connecting sign-in came back refused (`connect=` from the with-provider
 * route), or one this tab started and never finished.
 */
export function useConnectFallback() {
  const refused = useSearchParams().get("connect");
  const [abandoned, setAbandoned] = useState<ConnectingProvider | null>(null);

  // Session storage exists only in the browser, so the note is read after
  // mount. Back from the provider can instead bring the page back as it was
  // before the sign-in started, so it is read again then.
  useEffect(() => takeAbandoned(setAbandoned), []);
  usePageRestored(() => takeAbandoned(setAbandoned));

  return isConnectingProvider(refused) ? refused : abandoned;
}

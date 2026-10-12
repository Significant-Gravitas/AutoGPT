import { useEffect, useState } from "react";

export function useQuoteExpiry(expiresAt?: Date | string) {
  const [now, setNow] = useState(Date.now);
  const expiration = expiresAt ? new Date(expiresAt).getTime() : null;
  useEffect(() => {
    if (expiration == null) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [expiration]);
  const remaining = expiration == null ? null : expiration - now;
  return {
    expired: remaining != null && remaining <= 0,
    expiringSoon: remaining != null && remaining > 0 && remaining < 120000,
  };
}

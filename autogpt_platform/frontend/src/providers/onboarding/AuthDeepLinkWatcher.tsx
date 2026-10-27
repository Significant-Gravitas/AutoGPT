"use client";

import { sanitizeAuthNext } from "@/lib/auth-redirect";
import { useSearchParams } from "next/navigation";
import { useEffect } from "react";

interface Props {
  onChange: (hasPendingAuthDeepLink: boolean) => void;
}

// Rendered only on /login and /signup, so only those pages subscribe the
// onboarding flow to search-param writes: a `?next=` that appears or goes away
// without a pathname change re-runs the provider's init effect.
export function AuthDeepLinkWatcher({ onChange }: Props) {
  const searchParams = useSearchParams();
  const hasPendingAuthDeepLink =
    sanitizeAuthNext(searchParams.get("next")) !== null;

  useEffect(() => {
    onChange(hasPendingAuthDeepLink);
  }, [hasPendingAuthDeepLink, onChange]);

  return null;
}

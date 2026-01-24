"use client";

import { getHomepageRoute } from "@/lib/constants";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function Page() {
  const isChatEnabled = useGetFlag(Flag.CHAT);
  const router = useRouter();
  const homepageRoute = getHomepageRoute(isChatEnabled);
  const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";
  const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
  const isLaunchDarklyConfigured = envEnabled && Boolean(clientId);
  const isFlagReady =
    !isLaunchDarklyConfigured || typeof isChatEnabled === "boolean";

  useEffect(
    function redirectToHomepage() {
      if (!isFlagReady) return;
      router.replace(homepageRoute);
    },
    [homepageRoute, isFlagReady, router],
  );

  return null;
}

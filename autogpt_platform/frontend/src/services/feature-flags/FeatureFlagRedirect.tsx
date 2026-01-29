"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useLDClient } from "launchdarkly-react-client-sdk";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { environment } from "../environment";
import { Flag, useGetFlag } from "./use-get-flag";

interface FeatureFlagRedirectProps {
  flag: Flag;
  whenEnabled: string;
  whenDisabled: string;
}

export function FeatureFlagRedirect({
  flag,
  whenEnabled,
  whenDisabled,
}: FeatureFlagRedirectProps) {
  const router = useRouter();
  const flagValue = useGetFlag(flag);
  const ldEnabled = environment.areFeatureFlagsEnabled();
  const ldClient = useLDClient();
  const ldReady = Boolean(ldClient);
  const flagEnabled = Boolean(flagValue);

  useEffect(() => {
    if (!ldEnabled) {
      router.replace(whenDisabled);
      return;
    }

    // Wait for LaunchDarkly to initialize when enabled to prevent race conditions
    if (ldEnabled && !ldReady) return;

    router.replace(flagEnabled ? whenEnabled : whenDisabled);
  }, [ldReady, flagEnabled]);

  return <LoadingSpinner size="large" cover />;
}

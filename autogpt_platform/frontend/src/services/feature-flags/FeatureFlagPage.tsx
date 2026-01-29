"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useLDClient } from "launchdarkly-react-client-sdk";
import { useRouter } from "next/navigation";
import { ReactNode, useEffect } from "react";
import { environment } from "../environment";
import { Flag, useGetFlag } from "./use-get-flag";

interface FeatureFlagRedirectProps {
  flag: Flag;
  whenDisabled: string;
  children: ReactNode;
}

export function FeatureFlagPage({
  flag,
  whenDisabled,
  children,
}: FeatureFlagRedirectProps) {
  const router = useRouter();
  const flagValue = useGetFlag(flag);
  const ldClient = useLDClient();
  const ldEnabled = environment.areFeatureFlagsEnabled();
  const ldReady = Boolean(ldClient);
  const flagEnabled = Boolean(flagValue);

  useEffect(() => {
    // Wait for LaunchDarkly to initialize when enabled to prevent race conditions
    if (ldEnabled && !ldReady) return;

    if (!ldEnabled || !flagEnabled) {
      router.replace(whenDisabled);
    }
  }, [ldEnabled, ldReady, flagEnabled]);

  return !flagEnabled ? <LoadingSpinner size="large" cover /> : <>{children}</>;
}

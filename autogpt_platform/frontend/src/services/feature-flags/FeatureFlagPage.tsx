"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useLDClient } from "launchdarkly-react-client-sdk";
import { useRouter } from "next/navigation";
import { ReactNode, useEffect, useState } from "react";
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
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  const flagValue = useGetFlag(flag);
  const ldClient = useLDClient();
  const ldEnabled = environment.areFeatureFlagsEnabled();
  const ldReady = Boolean(ldClient);
  const flagEnabled = Boolean(flagValue);

  useEffect(() => {
    const initialize = async () => {
      if (!ldEnabled) {
        router.replace(whenDisabled);
        setIsLoading(false);
        return;
      }

      // Wait for LaunchDarkly to initialize when enabled to prevent race conditions
      if (ldEnabled && !ldReady) return;

      try {
        await ldClient?.waitForInitialization();
        if (!flagEnabled) router.replace(whenDisabled);
      } catch (error) {
        console.error(error);
        router.replace(whenDisabled);
      } finally {
        setIsLoading(false);
      }
    };

    initialize();
  }, [ldReady, flagEnabled]);

  return isLoading || !flagEnabled ? (
    <LoadingSpinner size="large" cover />
  ) : (
    <>{children}</>
  );
}

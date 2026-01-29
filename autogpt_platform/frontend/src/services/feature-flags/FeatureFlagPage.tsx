"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useLDClient } from "launchdarkly-react-client-sdk";
import { useRouter } from "next/navigation";
import { ReactNode, useEffect } from "react";
import { environment } from "../environment";
import { Flag, useGetFlag } from "./use-get-flag";

interface FeatureFlagRedirectProps {
  /** The feature flag to check */
  flag: Flag;
  /** Route to redirect to when flag is false */
  whenDisabled: string;
  /** Children to render when flag is enabled */
  children: ReactNode;
}

/**
 * Component that redirects based on a feature flag value.
 *
 * Waits for LaunchDarkly to initialize before redirecting to avoid
 * race conditions where the flag hasn't loaded yet.
 *
 * @example
 * ```tsx
 * export default function Page() {
 *   return (
 *     <FeatureFlagRedirect
 *       flag={Flag.CHAT}
 *       whenDisabled="/library"
 *     >
 *       <div>Feature is enabled!</div>
 *     </FeatureFlagGate>
 *     />
 *   );
 * }
 * ```
 */
export function FeatureFlagPage({
  flag,
  whenDisabled,
  children,
}: FeatureFlagRedirectProps) {
  const router = useRouter();
  const flagValue = useGetFlag(flag);
  const ldClient = useLDClient();
  const ldEnabled = environment.areFeatureFlagsEnabled();
  const ldReady = typeof flagValue !== "undefined" && Boolean(ldClient);
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

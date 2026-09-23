"use client";

import { useFlags } from "launchdarkly-react-client-sdk";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { envFlagOverride, Flag } from "./use-get-flag";

export function withFeatureFlag<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  flagKey: string,
) {
  return function FeatureFlaggedComponent(props: P) {
    const flags = useFlags();
    const router = useRouter();

    // The local env override (per-flag NEXT_PUBLIC_FORCE_FLAG_*, or the
    // NEXT_PUBLIC_FORCE_ALL_FLAGS master switch) wins over LaunchDarkly, so a
    // page gated by this HOC respects force-all like the useGetFlag hook does.
    const override = envFlagOverride(flagKey as Flag);
    const isEnabled = override !== undefined ? override : flags[flagKey];

    const [hasFlagLoaded, setHasFlagLoaded] = useState(override !== undefined);

    useEffect(() => {
      if (override !== undefined || (flags && flagKey in flags)) {
        setHasFlagLoaded(true);
      }
    }, [flags, override]);

    useEffect(() => {
      if (hasFlagLoaded && !isEnabled) {
        router.push("/404");
      }
    }, [hasFlagLoaded, isEnabled, router]);

    // Show loading state until flags loaded
    if (!hasFlagLoaded) {
      return (
        <div className="flex min-h-screen items-center justify-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      );
    }

    // If flag is loaded but false, return null (will redirect)
    if (!isEnabled) {
      return null;
    }

    // Flag is loaded and true, show component
    return <WrappedComponent {...props} />;
  };
}

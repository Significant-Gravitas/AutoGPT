"use client";

import { useFlags } from "launchdarkly-react-client-sdk";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export function withFeatureFlag<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  flagKey: string,
) {
  return function FeatureFlaggedComponent(props: P) {
    const flags = useFlags();
    const router = useRouter();
    const [hasFlagLoaded, setHasFlagLoaded] = useState(false);

    useEffect(() => {
      // Only proceed if flags received
      if (flags && flagKey in flags) {
        setHasFlagLoaded(true);
      }
    }, [flags]);

    useEffect(() => {
      if (hasFlagLoaded && !flags[flagKey]) {
        router.push("/404");
      }
    }, [hasFlagLoaded, flags, router]);

    // Show loading state until flags loaded
    if (!hasFlagLoaded) {
      return (
        <div className="flex min-h-screen items-center justify-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      );
    }

    // If flag is loaded but false, return null (will redirect)
    if (!flags[flagKey]) {
      return null;
    }

    // Flag is loaded and true, show component
    return <WrappedComponent {...props} />;
  };
}

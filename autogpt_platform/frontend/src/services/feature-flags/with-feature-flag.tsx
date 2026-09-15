"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { Flag, useFlagStatus } from "./use-get-flag";

export function withFeatureFlag<P extends object, T extends Flag>(
  WrappedComponent: React.ComponentType<P>,
  flag: T,
) {
  return function FeatureFlaggedComponent(props: P) {
    const { enabled, ready } = useFlagStatus(flag);
    const router = useRouter();

    useEffect(() => {
      if (ready && !enabled) {
        router.push("/404");
      }
    }, [ready, enabled, router]);

    // Reading through useFlagStatus rather than the LaunchDarkly SDK is what
    // bounds this wait: `ready` resolves on the vendor's answer, on the 5s
    // timeout, or immediately when no vendor is configured.
    if (!ready) {
      return (
        <div className="flex min-h-screen items-center justify-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      );
    }

    if (!enabled) {
      return null;
    }

    return <WrappedComponent {...props} />;
  };
}

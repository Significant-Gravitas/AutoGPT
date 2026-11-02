"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { Flag, useFlagStatus } from "./use-get-flag";

export function withFeatureFlag<P extends object, T extends Flag>(
  WrappedComponent: React.ComponentType<P>,
  flag: T,
) {
  return function FeatureFlaggedComponent(props: P) {
    const { enabled, answered } = useFlagStatus(flag);
    const router = useRouter();

    // Navigating on the 5s timeout would send a user who has the flag to
    // /404 whenever the vendor is slow or blocked, so only an answer moves.
    useEffect(() => {
      if (answered && !enabled) {
        router.push("/404");
      }
    }, [answered, enabled, router]);

    if (!answered) {
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

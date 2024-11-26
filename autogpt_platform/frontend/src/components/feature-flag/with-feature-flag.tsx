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
    const [isInitialized, setIsInitialized] = useState(false);
    const isEnabled = flags[flagKey];

    useEffect(() => {
      const timer = setTimeout(() => {
        setIsInitialized(true);
      }, 100);

      return () => clearTimeout(timer);
    }, []);

    useEffect(() => {
      if (isInitialized && !isEnabled) {
        router.push("/404");
      }
    }, [isInitialized, isEnabled, router]);

    if (!isInitialized) {
      return (
        <div className="flex min-h-screen items-center justify-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        </div>
      );
    }

    if (!isEnabled) {
      return null;
    }

    return <WrappedComponent {...props} />;
  };
}

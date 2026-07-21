"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import * as Sentry from "@sentry/nextjs";
import { LDProvider } from "launchdarkly-react-client-sdk";
import type { ReactNode } from "react";
import { useMemo } from "react";
import { environment } from "../environment";
import { buildLDContext } from "./helpers";

const LAUNCHDARKLY_INIT_TIMEOUT_MS = 5000;

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  const { user, isUserLoading } = useSupabase();
  const envEnabled = environment.areFeatureFlagsEnabled();
  const clientId = environment.getLaunchDarklyClientId();

  const context = useMemo(() => {
    if (isUserLoading) return;
    return buildLDContext(user);
  }, [user, isUserLoading]);

  if (!envEnabled) {
    return <>{children}</>;
  }

  if (isUserLoading) {
    return <LoadingSpinner size="large" cover />;
  }

  return (
    <LDProvider
      clientSideID={clientId ?? ""}
      context={context}
      timeout={LAUNCHDARKLY_INIT_TIMEOUT_MS}
      reactOptions={{ useCamelCaseFlagKeys: false }}
      options={{
        inspectors: [Sentry.buildLaunchDarklyFlagUsedHandler()],
      }}
    >
      {children}
    </LDProvider>
  );
}

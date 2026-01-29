"use client";

import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import * as Sentry from "@sentry/nextjs";
import { LDProvider } from "launchdarkly-react-client-sdk";
import type { ReactNode } from "react";
import { useMemo } from "react";
import { environment } from "../environment";

const LAUNCHDARKLY_INIT_TIMEOUT_MS = 5000;

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  const { user, isUserLoading } = useSupabase();
  const envEnabled = environment.areFeatureFlagsEnabled();
  const clientId = environment.getLaunchDarklyClientId();

  const context = useMemo(() => {
    if (isUserLoading || !user) {
      return {
        kind: "user" as const,
        key: "anonymous",
        anonymous: true,
      };
    }

    return {
      kind: "user" as const,
      key: user.id,
      ...(user.email && { email: user.email }),
      anonymous: false,
      custom: {
        ...(user.role && { role: user.role }),
      },
    };
  }, [user, isUserLoading]);

  if (!envEnabled) {
    return <>{children}</>;
  }

  return (
    <LDProvider
      // Add this key prop. It will be 'anonymous' when logged out,
      key={context.key}
      clientSideID={clientId ?? ""}
      context={context}
      timeout={LAUNCHDARKLY_INIT_TIMEOUT_MS}
      reactOptions={{ useCamelCaseFlagKeys: false }}
      options={{
        bootstrap: "localStorage",
        inspectors: [Sentry.buildLaunchDarklyFlagUsedHandler()],
      }}
    >
      {children}
    </LDProvider>
  );
}

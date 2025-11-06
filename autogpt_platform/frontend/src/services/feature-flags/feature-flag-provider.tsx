"use client";

import { LDProvider } from "launchdarkly-react-client-sdk";
import type { ReactNode } from "react";
import { useMemo } from "react";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import * as Sentry from "@sentry/nextjs";
import { environment } from "../environment";

const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  const { user, isUserLoading } = useSupabase();
  const isCloud = environment.isCloud();
  const isLaunchDarklyConfigured = isCloud && envEnabled && clientId;

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

  if (!isLaunchDarklyConfigured) {
    return <>{children}</>;
  }

  return (
    <LDProvider
      // Add this key prop. It will be 'anonymous' when logged out,
      key={context.key}
      clientSideID={clientId}
      context={context}
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

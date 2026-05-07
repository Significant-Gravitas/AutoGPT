"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
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
    if (isUserLoading) return;

    if (!user) {
      return {
        kind: "user" as const,
        key: "anonymous",
        anonymous: true,
      };
    }

    // Mirror the context built by the backend
    // (feature_flag.py:_fetch_user_context_data) so LaunchDarkly targeting
    // rules evaluate identically on both sides.
    return {
      kind: "user" as const,
      key: user.id,
      anonymous: false,
      ...(user.email && {
        email: user.email,
        email_domain: user.email.split("@").at(-1),
      }),
      ...(user.role && { role: user.role }),
      // Supabase JS emits `Z`-suffixed ISO; backend emits `+00:00` — LD date matchers accept both.
      ...(user.created_at && { created_at: user.created_at }),
      custom: {
        ...(user.role && { role: user.role }),
      },
    };
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

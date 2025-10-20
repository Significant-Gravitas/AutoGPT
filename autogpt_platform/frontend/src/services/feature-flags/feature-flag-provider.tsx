"use client";

import { LDProvider } from "launchdarkly-react-client-sdk";
import type { ReactNode } from "react";
import { useMemo } from "react";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { BehaveAs, getBehaveAs } from "@/lib/utils";
import * as Sentry from "@sentry/nextjs";

const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";

/**
 * Creates a defensive wrapper around Sentry's LaunchDarkly flag handler
 * to catch and prevent any errors from breaking the application.
 */
function createSafeLaunchDarklyFlagHandler() {
  try {
    const handler = Sentry.buildLaunchDarklyFlagUsedHandler();
    
    // Wrap the handler to catch any runtime errors
    return (flagKey: string, detail: any) => {
      try {
        handler(flagKey, detail);
      } catch (error) {
        // Log the error to console but don't let it bubble up
        console.error("Error in Sentry LaunchDarkly flag handler:", error);
      }
    };
  } catch (error) {
    // If building the handler fails, return a no-op function
    console.error("Failed to build Sentry LaunchDarkly flag handler:", error);
    return () => {};
  }
}

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  const { user, isUserLoading } = useSupabase();
  const isCloud = getBehaveAs() === BehaveAs.CLOUD;
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
        inspectors: [createSafeLaunchDarklyFlagHandler()],
      }}
    >
      {children}
    </LDProvider>
  );
}

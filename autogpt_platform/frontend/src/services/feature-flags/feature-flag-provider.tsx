"use client";

import { LDProvider } from "launchdarkly-react-client-sdk";
import type { ReactNode } from "react";
import { useMemo } from "react";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { BehaveAs, getBehaveAs } from "@/lib/utils";

const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  const { user, isUserLoading } = useSupabase();
  const isCloud = getBehaveAs() === BehaveAs.CLOUD;
  const isLaunchDarklyConfigured = isCloud && envEnabled && clientId;

  // Create the context memoized, and it will change reactively based on the user's authentication state
  const ldContext = useMemo(() => {
    // While loading, or if there's no authenticated user, use an anonymous context
    if (isUserLoading || !user) {
      console.log("[LaunchDarklyProvider] Using anonymous context", {
        isUserLoading,
        hasUser: !!user,
      });
      return {
        kind: "user" as const,
        key: "anonymous",
        anonymous: true,
      };
    }

    // Once the user is loaded, create the authenticated context
    console.log("[LaunchDarklyProvider] Using authenticated context", {
      userId: user.id,
      email: user.email,
      role: user.role,
    });
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

  // If LaunchDarkly isn't configured for this environment, don't render the provider
  if (!isLaunchDarklyConfigured) {
    console.log("[LaunchDarklyProvider] Not configured for this environment", {
      isCloud,
      envEnabled,
      hasClientId: !!clientId,
    });
    return <>{children}</>;
  }

  // The LDProvider is now always rendered (if configured), but its context
  // prop will change from anonymous to authenticated, triggering the SDK to update
  return (
    <LDProvider
      // Add this key prop. It will be 'anonymous' when logged out,
      // and the user's ID when logged in.
      key={ldContext.key}
      clientSideID={clientId}
      context={ldContext}
      reactOptions={{ useCamelCaseFlagKeys: false }}
      options={{ bootstrap: "localStorage" }}
    >
      {children}
    </LDProvider>
  );
}

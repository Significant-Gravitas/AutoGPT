import { LDProvider, useLDClient } from "launchdarkly-react-client-sdk";
import { ReactNode, useEffect } from "react";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { BehaveAs, getBehaveAs } from "@/lib/utils";
// import { getServerUser } from "@/lib/supabase/server/getServerUser";

const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";

// Inner component that uses the LD client to update user context
function LaunchDarklyUpdater({ children }: { children: ReactNode }) {
  const { user, isUserLoading } = useSupabase();
  const ldClient = useLDClient();

  useEffect(() => {
    console.log("[LaunchDarklyUpdater] Effect triggered", {
      isUserLoading,
      hasUser: !!user,
      userId: user?.id,
      userEmail: user?.email,
      hasLdClient: !!ldClient,
    });

    // When user loading completes and we have a user, identify them
    if (!isUserLoading && user && ldClient) {
      console.log("[LaunchDarklyUpdater] Identifying user in LaunchDarkly:", {
        userId: user.id,
        email: user.email,
        role: user.role,
      });

      const userContext = {
        kind: "user" as const,
        key: user.id,
        email: user.email,
        anonymous: false,
        custom: {
          role: user.role,
        },
      };

      ldClient
        .identify(userContext)
        .then(() => {
          console.log("[LaunchDarklyUpdater] User identified successfully");
        })
        .catch((error) => {
          console.error(
            "[LaunchDarklyUpdater] Failed to identify user:",
            error,
          );
        });
    } else if (!isUserLoading && !user && ldClient) {
      console.log(
        "[LaunchDarklyUpdater] User loading complete but no user found, staying anonymous",
      );
    }
  }, [user, isUserLoading, ldClient]);

  return <>{children}</>;
}

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  console.log("[LaunchDarklyProvider] Render started");
  const { user: userS, supabase, isUserLoading } = useSupabase();
  const isCloud = getBehaveAs() === BehaveAs.CLOUD;
  const enabled = isCloud && envEnabled && clientId;

  // Log current state
  console.log("[LaunchDarklyProvider] Current state:", {
    enabled,
    isCloud,
    envEnabled,
    clientId: clientId ? `${clientId.substring(0, 8)}...` : undefined,
    isUserLoading,
    hasUser: !!userS,
    userId: userS?.id,
    userEmail: userS?.email,
    userRole: userS?.role,
  });

  // Async check for user from supabase
  supabase?.auth.getUser().then(({ data: { user: userR }, error }) => {
    console.log("[LaunchDarklyProvider] Supabase auth.getUser result:", {
      hasUser: !!userR,
      userId: userR?.id,
      userEmail: userR?.email,
      error: error?.message,
    });
  });

  // If LaunchDarkly is not enabled for this environment, just render children
  if (!enabled) {
    console.log(
      "[LaunchDarklyProvider] Not enabled, rendering children without LD",
    );
    return <>{children}</>;
  }

  // Always start with anonymous context
  // The LaunchDarklyUpdater will identify the user once they're loaded
  const initialContext = {
    kind: "user" as const,
    key: "anonymous",
    anonymous: true,
  };

  console.log(
    "[LaunchDarklyProvider] Initializing with anonymous context:",
    initialContext,
  );

  return (
    <LDProvider
      clientSideID={clientId}
      context={initialContext}
      reactOptions={{ useCamelCaseFlagKeys: false }}
      options={{
        bootstrap: "localStorage",
      }}
    >
      <LaunchDarklyUpdater>{children}</LaunchDarklyUpdater>
    </LDProvider>
  );
}

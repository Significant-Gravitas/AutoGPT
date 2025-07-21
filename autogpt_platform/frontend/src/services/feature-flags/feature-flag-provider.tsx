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
    // When user loading completes and we have a user, identify them
    if (!isUserLoading && user && ldClient) {
      console.log("Identifying user in LaunchDarkly:", user.id);
      const userContext = {
        kind: "user" as const,
        key: user.id,
        email: user.email,
        anonymous: false,
        custom: {
          role: user.role,
        },
      };
      ldClient.identify(userContext);
    }
  }, [user, isUserLoading, ldClient]);

  return <>{children}</>;
}

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  console.log("LaunchDarklyProvider render");
  const { user: userS, supabase } = useSupabase();
  const isCloud = getBehaveAs() === BehaveAs.CLOUD;
  const enabled = isCloud && envEnabled && clientId;

  supabase?.auth.getUser().then(({ data: { user: userR } }) => {
    console.log(`user from supabase ${userR}`);
  });

  console.log(
    `ld status ${enabled} iscloud ${isCloud} envEnabled ${envEnabled} clientId ${clientId} user server ${userS} `,
  );

  // If LaunchDarkly is not enabled for this environment, just render children
  if (!enabled) return <>{children}</>;

  // Always start with anonymous context
  // The LaunchDarklyUpdater will identify the user once they're loaded
  const initialContext = {
    kind: "user" as const,
    key: "anonymous",
    anonymous: true,
  };

  console.log(`initial context ${JSON.stringify(initialContext)}`);

  return (
    <LDProvider
      clientSideID={clientId}
      context={initialContext}
      reactOptions={{ useCamelCaseFlagKeys: false }}
    >
      <LaunchDarklyUpdater>{children}</LaunchDarklyUpdater>
    </LDProvider>
  );
}

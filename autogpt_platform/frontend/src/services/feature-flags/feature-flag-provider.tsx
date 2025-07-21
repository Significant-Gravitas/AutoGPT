import { LDProvider } from "launchdarkly-react-client-sdk";
import { ReactNode } from "react";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { BehaveAs, getBehaveAs } from "@/lib/utils";
// import { getServerUser } from "@/lib/supabase/server/getServerUser";

const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  console.log("LaunchDarklyProvider render");
  // const { user: userS } = await getServerUser();
  const { user: userS, supabase } = useSupabase();
  const isCloud = getBehaveAs() === BehaveAs.CLOUD;
  const enabled = isCloud && envEnabled && clientId && userS;
  const user = userS;

  supabase?.auth.getUser().then(({ data: { user: userR } }) => {
    console.log(`user from supabase ${userR}`);
  });

  console.log(
    `ld status ${enabled} iscloud ${isCloud} envEnabled ${envEnabled} clientId ${clientId} user server ${userS} `,
  );

  if (!enabled) return <>{children}</>;

  const userContext = user
    ? {
        kind: "user",
        key: user.id,
        email: user.email,
        anonymous: false,
        custom: {
          role: user.role,
        },
      }
    : {
        kind: "user",
        key: "anonymous",
        anonymous: true,
      };
  console.log(`user context ${userContext}`);

  return (
    <LDProvider
      clientSideID={clientId}
      context={userContext}
      reactOptions={{ useCamelCaseFlagKeys: false }}
    >
      {children}
    </LDProvider>
  );
}

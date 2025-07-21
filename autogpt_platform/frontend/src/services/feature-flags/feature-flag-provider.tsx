import { LDProvider } from "launchdarkly-react-client-sdk";
import { ReactNode } from "react";
// import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { BehaveAs, getBehaveAs } from "@/lib/utils";
import { getServerUser } from "@/lib/supabase/server/getServerUser";
import { getCurrentUser } from "@/lib/supabase/actions";

const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";

export async function LaunchDarklyProvider({
  children,
}: {
  children: ReactNode;
}) {
  console.log("LaunchDarklyProvider render");
  const { user: userS } = await getServerUser();
  const { user: userC } = await getCurrentUser();
  const isCloud = getBehaveAs() === BehaveAs.CLOUD;
  const enabled = isCloud && envEnabled && clientId && userS && userC;
  const user = userS;

  console.log(
    `ld status ${enabled} iscloud ${isCloud} envEnabled ${envEnabled} clientId ${clientId} user ${userS} userC ${userC}`,
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

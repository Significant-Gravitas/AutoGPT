"use client";

import { LDProvider } from "launchdarkly-react-client-sdk";
import { ReactNode } from "react";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { BehaveAs, getBehaveAs } from "@/lib/utils";

const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
const envEnabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  const { user } = useSupabase();
  const isCloud = getBehaveAs() === BehaveAs.CLOUD;
  const enabled = isCloud && envEnabled && clientId && user;

  console.log(`ld status ${enabled} iscloud ${isCloud}`);

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

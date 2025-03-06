import { LDProvider } from "launchdarkly-react-client-sdk";
import { ReactNode } from "react";

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
  const enabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";

  if (!enabled) return <>{children}</>;

  if (!clientId) {
    throw new Error("NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID is not defined");
  }

  return <LDProvider clientSideID={clientId}>{children}</LDProvider>;
}

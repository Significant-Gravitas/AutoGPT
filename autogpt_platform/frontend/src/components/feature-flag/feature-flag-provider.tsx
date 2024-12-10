import { LDProvider } from "launchdarkly-react-client-sdk";
import { ReactNode } from "react";

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  if (
    process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === true &&
    !process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID
  ) {
    throw new Error("NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID is not defined");
  }

  return (
    <LDProvider clientSideID={process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID}>
      {children}
    </LDProvider>
  );
}

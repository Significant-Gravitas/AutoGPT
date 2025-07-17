import { LDProvider } from "launchdarkly-react-client-sdk";
import { ReactNode } from "react";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  const clientId = process.env.NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID;
  const enabled = process.env.NEXT_PUBLIC_LAUNCHDARKLY_ENABLED === "true";
  const { user, isUserLoading } = useSupabase();

  if (!enabled) return <>{children}</>;

  if (!clientId) {
    throw new Error("NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID is not defined");
  }

  // Show loading state while user is being determined
  if (isUserLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  // Create user context for LaunchDarkly
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

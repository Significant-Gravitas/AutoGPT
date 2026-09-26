"use client";

import { useAuth } from "@/lib/auth/hooks/useAuth";
import { LDProvider } from "launchdarkly-react-client-sdk";
import type { ReactNode } from "react";
import { useMemo } from "react";
import { getAnonymousID } from "../analytics/anonymous-id";
import { environment } from "../environment";
import { LD_INIT_TIMEOUT_SECONDS } from "./constants";
import { usesLaunchDarkly } from "./flag-backend";
import { FlagResolutionContext } from "./flag-resolution";
import { buildLDContext } from "./helpers";

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  const { user, isUserLoading } = useAuth();
  const envEnabled = usesLaunchDarkly() && environment.areFeatureFlagsEnabled();
  const clientId = environment.getLaunchDarklyClientId();

  // Undefined until the session check resolves, which it never does on the
  // server. The page renders regardless so crawlers get its content; the
  // provider below just holds off initialising until the context is known,
  // during which every flag reads as "not answered yet" and the resolution
  // timeout in ``useFlagStatus`` has not started counting.
  const context = useMemo(() => {
    if (isUserLoading) return;
    return buildLDContext(user, getAnonymousID());
  }, [user, isUserLoading]);

  if (!envEnabled) {
    return <>{children}</>;
  }

  return (
    <FlagResolutionContext.Provider value={context !== undefined}>
      <LDProvider
        clientSideID={clientId ?? ""}
        context={context}
        deferInitialization
        timeout={LD_INIT_TIMEOUT_SECONDS}
        reactOptions={{ useCamelCaseFlagKeys: false }}
      >
        {children}
      </LDProvider>
    </FlagResolutionContext.Provider>
  );
}

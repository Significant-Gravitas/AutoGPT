import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { filterSystemCredentials } from "@/components/contextual/CredentialsInput/helpers";
import { getAppNames, toDisconnectTargets } from "./context-helpers";
import type { AppsState } from "./helpers";

export function useConnectedApps() {
  const { data: credentials } = useGetV1ListCredentials({
    query: {
      select: (response) =>
        response.status === 200
          ? filterSystemCredentials(response.data).filter(
              (credential) => !credential.is_managed,
            )
          : undefined,
    },
  });

  const apps: AppsState = credentials
    ? { status: "ready", names: getAppNames(credentials) }
    : { status: "unavailable" };

  return { apps, targets: toDisconnectTargets(credentials ?? []) };
}

import {
  getListExpertCredentialsQueryKey,
  useListExpertCredentials,
  type grantExpertCredentialsResponse,
  type listExpertCredentialsResponseSuccess,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { findSavedUserCredentialByProviderAndType } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import type { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import type { ConnectorRow } from "./helpers";

export function useExpertCredentialSelection(
  row: ConnectorRow,
  providers: CredentialsProvidersContextType | null,
) {
  const queryClient = useQueryClient();
  const expertId = row.expertGrant?.expertId;
  const grants = useListExpertCredentials(expertId ?? "", {
    query: {
      enabled: Boolean(row.expertGrant),
      select: (response) => okData(response),
    },
  });
  const provider = providers?.[row.provider];
  const grantedIDs = new Set(grants.data?.map((grant) => grant.credential_id));
  const saved = provider?.savedCredentials.filter((credential) =>
    grantedIDs.has(credential.id),
  );
  function findCredential(id?: string) {
    if (!provider || !saved) return undefined;
    return findSavedUserCredentialByProviderAndType(
      row.schema.credentials_provider ?? [],
      row.schema.credentials_types ?? [],
      row.schema.credentials_scopes,
      {
        [row.provider]: {
          ...provider,
          savedCredentials: id
            ? saved.filter((credential) => credential.id === id)
            : saved,
        },
      },
      row.schema.discriminator_values,
    );
  }
  // The matcher abstains when several accounts qualify, which would leave the
  // row on Connect with an empty offer — the backend omits already-granted
  // accounts — and no way forward. Grant order breaks the tie.
  function firstGrantedMatch() {
    for (const grant of grants.data ?? []) {
      const match = findCredential(grant.credential_id);
      if (match) return match;
    }
    return undefined;
  }

  const selectedID = row.selected?.id;
  const isSelectedGranted = Boolean(selectedID && grantedIDs.has(selectedID));
  // The provider list reloads asynchronously after a connect, so an account
  // it has not caught up with is merely absent, not unusable.
  const isSelectedLoaded = Boolean(
    selectedID &&
      provider?.savedCredentials.some(
        (credential) => credential.id === selectedID,
      ),
  );
  // A granted selection stays put: swapping it for a different granted
  // account would undo the choice just made, and clearing it would flip the
  // row from Granted back to Connect until the provider list catches up. Only
  // a revoked grant, or a loaded account that fails the row, drops it.
  const keepsSelection =
    isSelectedGranted &&
    (!isSelectedLoaded || Boolean(findCredential(selectedID)));
  const hydrated = keepsSelection ? undefined : firstGrantedMatch();
  const hasGrant = Boolean(row.expertGrant);
  const isSelectionGranted = Boolean(
    hasGrant && !grants.isError && grants.data && keepsSelection,
  );

  useEffect(() => {
    if (!hasGrant || !grants.data || !providers || grants.isFetching) return;
    if (keepsSelection || selectedID === hydrated?.id) return;
    row.select(
      hydrated
        ? {
            id: hydrated.id,
            provider: row.provider,
            type: hydrated.type as CredentialsMetaInput["type"],
            title: hydrated.title ?? undefined,
          }
        : undefined,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps -- row is rebuilt each render by the card; track the fields it reads
  }, [
    hasGrant,
    selectedID,
    hydrated?.id,
    keepsSelection,
    grants.data,
    grants.isFetching,
    providers,
  ]);

  return {
    isPending: grants.isPending,
    isError: grants.isError,
    isSelectionGranted,
    /** Records the list the POST returned as the expert's grants and reports
     *  whether it holds `id`. That response is the only proof available: the
     *  list query is shared by every row of this expert, so a refetch started
     *  here is cancelled by a sibling row's and resolves against pre-POST
     *  data — a grant that landed would read back as a failure. */
    confirmGrant(id: string, response: grantExpertCredentialsResponse) {
      const granted = okData(response);
      if (!granted || !expertId) return false;
      const refreshed: listExpertCredentialsResponseSuccess = {
        status: 200,
        data: granted,
        headers: response.headers,
      };
      queryClient.setQueryData(
        getListExpertCredentialsQueryKey(expertId),
        refreshed,
      );
      return granted.some((grant) => grant.credential_id === id);
    },
  };
}

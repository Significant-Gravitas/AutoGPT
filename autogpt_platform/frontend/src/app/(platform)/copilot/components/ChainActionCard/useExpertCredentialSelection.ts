import { useListExpertCredentials } from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { findSavedUserCredentialByProviderAndType } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import type { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";
import { useEffect } from "react";
import type { ConnectorRow } from "./helpers";

export function useExpertCredentialSelection(
  row: ConnectorRow,
  providers: CredentialsProvidersContextType | null,
) {
  const grants = useListExpertCredentials(row.expertGrant?.expertId ?? "", {
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
  const credential = findCredential(row.selected?.id) ?? findCredential();
  const hasGrant = Boolean(row.expertGrant);

  useEffect(() => {
    if (!hasGrant || !grants.data || !providers || grants.isFetching) return;
    if (row.selected?.id === credential?.id) return;
    row.select(
      credential
        ? {
            id: credential.id,
            provider: row.provider,
            type: credential.type as CredentialsMetaInput["type"],
            title: credential.title ?? undefined,
          }
        : undefined,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps -- row is rebuilt each render by the card; track the fields it reads
  }, [
    hasGrant,
    row.selected?.id,
    credential?.id,
    grants.data,
    grants.isFetching,
    providers,
  ]);

  return grants;
}

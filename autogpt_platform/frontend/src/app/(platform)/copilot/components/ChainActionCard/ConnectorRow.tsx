"use client";

import { useGrantExpertCredentials } from "@/app/api/__generated__/endpoints/experts/experts";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ConnectCredentialDialog } from "@/components/contextual/CredentialsInput/components/ConnectCredentialDialog/ConnectCredentialDialog";
import { findSavedUserCredentialByProviderAndType } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";
import { filterSystemCredentials } from "@/components/contextual/CredentialsInput/helpers";
import { ProviderAvatar } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/ProviderAvatar";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import {
  CredentialsProvidersContext,
  type CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { useContext, useEffect, useState } from "react";
import type { ConnectorRow as Row } from "./helpers";

interface Props {
  row: Row;
}

export function ConnectorRow({ row }: Props) {
  const [isDialogOpen, setDialogOpen] = useState(false);
  const [awaitingGrant, setAwaitingGrant] = useState(false);
  const [grantError, setGrantError] = useState<string | null>(null);
  const allProviders = useContext(CredentialsProvidersContext);
  const { mutateAsync: grantCredentials, isPending: isGranting } =
    useGrantExpertCredentials();
  const expertGrant = row.expertGrant;

  // A credential the user already had — or one they just created in the
  // dialog — satisfies this row, so pick it up as soon as the providers
  // query refreshes rather than making them choose it again.
  const savedCredential = findSavedUserCredentialByProviderAndType(
    row.schema.credentials_provider ?? [],
    row.schema.credentials_types ?? [],
    row.schema.credentials_scopes,
    allProviders,
    row.schema.discriminator_values,
  );

  async function grant(credential: {
    id: string;
    provider: string;
    type: string;
    title?: string | null;
  }) {
    if (!expertGrant) return;
    setGrantError(null);
    try {
      await grantCredentials({
        expertId: expertGrant.expertId,
        data: { credential_ids: [credential.id] },
      });
    } catch {
      setGrantError("Couldn't grant access. Try again.");
      return;
    }
    row.select({
      id: credential.id,
      provider: credential.provider,
      type: credential.type as CredentialsMetaInput["type"],
      title: credential.title ?? undefined,
    });
    row.onConnected();
  }

  useEffect(() => {
    // An expert's row: the account having a credential is not enough, the
    // expert must be granted it. A credential the user just connected in the
    // dialog is granted here as soon as the providers query surfaces it.
    if (expertGrant) {
      if (awaitingGrant && savedCredential && !row.selected) {
        setAwaitingGrant(false);
        void grant(savedCredential);
      }
      return;
    }
    // Cards stream in one commit at a time, so a row's schema can widen after
    // it auto-selected: a selection that no longer satisfies it must go, or
    // the row reads Connected while Proceed sends a credential missing the
    // scopes the later card asked for. `null` is the provider context's
    // "still loading" sentinel, where every lookup misses — clearing then
    // would drop a good selection on every mount.
    if (allProviders && row.selected && !savedCredential) {
      row.select(undefined);
      return;
    }
    if (row.selected || !savedCredential) return;
    row.select({
      id: savedCredential.id,
      provider: savedCredential.provider,
      type: savedCredential.type as CredentialsMetaInput["type"],
      title: savedCredential.title ?? undefined,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- row.select is rebuilt each render by the card
  }, [savedCredential?.id, row.selected, allProviders, awaitingGrant]);

  const grantable = expertGrant?.credentials[0];

  return (
    <div className="flex items-center gap-3 px-4 py-3">
      <span className="flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-2xl border border-zinc-100 bg-white p-1.5">
        <ProviderAvatar id={row.provider} name={row.displayName} />
      </span>

      <div className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-sm font-medium text-zinc-900">
          {row.displayName}
        </span>
        {grantError ? (
          <span className="truncate text-sm text-red-600">{grantError}</span>
        ) : expertGrant && !row.selected ? (
          <span className="truncate text-sm text-zinc-500">
            {grantable
              ? `Your account has ${grantable.title}; this expert needs access to it`
              : "This expert needs its own access to this integration"}
          </span>
        ) : (
          row.description && (
            <span className="truncate text-sm text-zinc-500">
              {row.description}
            </span>
          )
        )}
      </div>

      {row.selected ? (
        <span className="flex shrink-0 items-center gap-1.5 text-sm font-medium text-zinc-500">
          <Icon icon={CheckmarkCircle02Icon} size={16} />
          {expertGrant ? "Granted" : "Connected"}
        </span>
      ) : grantable ? (
        <Button
          variant="primary"
          size="small"
          className="shrink-0"
          disabled={isGranting}
          onClick={() =>
            grant({
              id: grantable.id,
              provider: row.provider,
              type: grantable.type,
              title: grantable.title,
            })
          }
        >
          Grant access
        </Button>
      ) : (
        <Button
          variant="primary"
          size="small"
          className="shrink-0"
          disabled={isGranting}
          onClick={() => setDialogOpen(true)}
        >
          Connect
        </Button>
      )}

      <ConnectCredentialDialog
        schema={row.schema}
        provider={row.provider}
        displayName={row.displayName}
        credentialID={upgradableCredentialID(row.provider, allProviders)}
        open={isDialogOpen}
        onClose={() => setDialogOpen(false)}
        onConnected={() => {
          if (expertGrant) {
            setAwaitingGrant(true);
            return;
          }
          row.onConnected();
        }}
      />
    </div>
  );
}

/** The account a re-auth should upgrade in place. Signing in without it can
 *  grant narrower scopes than the user already had and leave a second row for
 *  the same provider, which no ConnectorRow can ever resolve. Only safe when
 *  exactly one account exists — otherwise picking one would be a guess.
 *  Managed and system credentials are excluded: the backend refuses to upgrade
 *  either, so offering one turns every Connect click into a 400. */
function upgradableCredentialID(
  provider: string,
  allProviders: CredentialsProvidersContextType | null,
) {
  const oauthCredentials = filterSystemCredentials(
    allProviders?.[provider]?.savedCredentials ?? [],
  ).filter(
    (credential) => credential.type === "oauth2" && !credential.is_managed,
  );
  return oauthCredentials.length === 1 ? oauthCredentials[0].id : undefined;
}

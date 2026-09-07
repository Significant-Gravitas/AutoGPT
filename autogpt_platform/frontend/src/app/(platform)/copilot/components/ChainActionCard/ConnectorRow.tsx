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
import { useContext, useEffect, useRef, useState } from "react";
import type { ConnectorRow as Row } from "./helpers";
import type { ExpertGrant } from "../SetupRequirementsCard/helpers";

interface Props {
  row: Row;
}

type Grantable = ExpertGrant["credentials"][number];

export function ConnectorRow({ row }: Props) {
  const [isDialogOpen, setDialogOpen] = useState(false);
  const [awaitingGrant, setAwaitingGrant] = useState(false);
  const [grantError, setGrantError] = useState<string | null>(null);
  // A credential connected from this row while an expert is asking. It stays
  // here until its grant succeeds, so a failed grant is retried from the
  // dialog's existing accounts instead of forcing another sign-in.
  const [connected, setConnected] = useState<Grantable | null>(null);
  // Credential ids that existed when Connect was clicked, so the grant goes
  // to the account the user just added rather than one they already had.
  const knownIds = useRef<Set<string>>(new Set());
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

  async function grant(credential: Grantable): Promise<boolean> {
    if (!expertGrant) return false;
    setGrantError(null);
    try {
      await grantCredentials({
        expertId: expertGrant.expertId,
        data: { credential_ids: [credential.id] },
      });
    } catch {
      setGrantError("Couldn't grant access. Try again.");
      return false;
    }
    setConnected(null);
    row.select({
      id: credential.id,
      provider: row.provider,
      type: credential.type as CredentialsMetaInput["type"],
      title: credential.title,
    });
    row.onConnected();
    return true;
  }

  useEffect(() => {
    if (expertGrant) {
      // The expert must be granted the credential; the account merely having
      // one is not enough. After a sign-in from this row, grant the account
      // that appeared since Connect was clicked — never a pre-existing one.
      if (!awaitingGrant || row.selected) return;
      const fresh = newlyConnectedCredential(
        row,
        allProviders,
        knownIds.current,
      );
      if (!fresh) return;
      setAwaitingGrant(false);
      setConnected(fresh);
      void grant(fresh);
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

  const grantableOptions = [
    ...(connected ? [connected] : []),
    ...(expertGrant?.credentials ?? []).filter((c) => c.id !== connected?.id),
  ];

  function openDialog() {
    knownIds.current = new Set(
      (allProviders?.[row.provider]?.savedCredentials ?? []).map((c) => c.id),
    );
    setDialogOpen(true);
  }

  return (
    <div className="flex items-center gap-3 px-4 py-2.5">
      <span className="flex size-9 shrink-0 items-center justify-center overflow-hidden rounded-2xl border border-zinc-100 bg-white p-1.5">
        <ProviderAvatar id={row.provider} name={row.displayName} />
      </span>

      <div className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-sm font-medium text-zinc-900">
          {row.displayName}
        </span>
        {grantError && !isDialogOpen ? (
          <span className="truncate text-xs text-red-600">{grantError}</span>
        ) : expertGrant && !row.selected ? (
          <span className="truncate text-xs text-zinc-500">
            {grantableOptions.length > 0
              ? "Needs this expert's access"
              : "This expert needs its own access"}
          </span>
        ) : (
          row.description && (
            <span className="truncate text-xs text-zinc-500">
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
      ) : (
        <Button
          variant="primary"
          size="small"
          className="shrink-0"
          disabled={isGranting}
          onClick={openDialog}
        >
          Connect
        </Button>
      )}

      <ConnectCredentialDialog
        schema={row.schema}
        provider={row.provider}
        displayName={row.displayName}
        credentialID={upgradableCredentialID(row.provider, allProviders)}
        existing={
          expertGrant
            ? {
                credentials: grantableOptions,
                onUse: grant,
                isPending: isGranting,
                error: grantError,
              }
            : undefined
        }
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

/** The credential that satisfies `row` and did not exist before Connect was
 *  clicked, i.e. the account the user just signed in with. */
function newlyConnectedCredential(
  row: Row,
  allProviders: CredentialsProvidersContextType | null,
  knownIds: Set<string>,
): Grantable | null {
  const provider = allProviders?.[row.provider];
  if (!provider) return null;
  const fresh = provider.savedCredentials.filter((c) => !knownIds.has(c.id));
  if (fresh.length === 0) return null;
  const match = findSavedUserCredentialByProviderAndType(
    row.schema.credentials_provider ?? [],
    row.schema.credentials_types ?? [],
    row.schema.credentials_scopes,
    { [row.provider]: { ...provider, savedCredentials: fresh } },
    row.schema.discriminator_values,
  );
  return match
    ? { id: match.id, title: match.title ?? row.displayName, type: match.type }
    : null;
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

"use client";

import { useGrantExpertCredentials } from "@/app/api/__generated__/endpoints/experts/experts";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ConnectCredentialDialog } from "@/components/contextual/CredentialsInput/components/ConnectCredentialDialog/ConnectCredentialDialog";
import { findSavedUserCredentialByProviderAndType } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";
import { filterSystemCredentials } from "@/components/contextual/CredentialsInput/helpers";
import { ProviderAvatar } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/ProviderAvatar";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import {
  CredentialsProvidersContext,
  type CredentialsProviderData,
  type CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { useContext, useEffect, useRef, useState } from "react";
import type { ConnectorRow as Row } from "./helpers";
import { useExpertCredentialSelection } from "./useExpertCredentialSelection";
import type { ExpertGrant } from "../SetupRequirementsCard/helpers";

interface Props {
  row: Row;
}

type Grantable = ExpertGrant["credentials"][number];
type SavedCredential = CredentialsProviderData["savedCredentials"][number];

const UNUSABLE_ACCOUNT_ERROR =
  "That account is missing the access this needs. Try connecting again.";
const GRANT_FAILED_ERROR = "Couldn't grant access. Try again.";

export function ConnectorRow({ row }: Props) {
  const [isDialogOpen, setDialogOpen] = useState(false);
  // A sign-in finished without reporting its credential, so the account it
  // added is found by diffing the provider list against `knownIds`.
  const [awaitingNewAccount, setAwaitingNewAccount] = useState(false);
  const [grantError, setGrantError] = useState<string | null>(null);
  // A credential connected from this row while an expert is asking. It stays
  // here until its grant succeeds, so a failed grant is retried from the
  // dialog's existing accounts instead of forcing another sign-in.
  const [connected, setConnected] = useState<Grantable | null>(null);
  // Credential ids that existed when Connect was clicked, so the grant goes
  // to the account the user just added rather than one they already had.
  // `null` while the provider's accounts were still loading, where a diff
  // would call every account the user already had new.
  const knownIds = useRef<Set<string> | null>(null);
  // Credentials the user signed in with from this row, latest last. A re-auth
  // updates the account in place and keeps its id, so the id alone cannot tell
  // a renewed credential from the one the provider refused.
  const [renewedIds, setRenewedIds] = useState<string[]>([]);
  const allProviders = useContext(CredentialsProvidersContext);
  const { mutateAsync: grantCredentials, isPending: isGranting } =
    useGrantExpertCredentials();
  const expertGrant = row.expertGrant;
  const grantedCredentials = useExpertCredentialSelection(row, allProviders);

  const rejectedIds = row.rejectedCredentialIds.filter(
    (id) => !renewedIds.includes(id),
  );
  // Still on file, but refused: never one to select or offer as usable.
  const usableProviders = withoutCredentials(allProviders, rejectedIds);
  // Only this provider's own refused account makes the row a Reconnect. A
  // card's rejection reaches every row it asks for, and the account may have
  // been deleted since. While one is on file nothing is picked for the user,
  // not even after a sign-in whose account the list has yet to show: quietly
  // running on another of their accounts could post as someone else.
  const onFile = (ids: string[]) =>
    !expertGrant &&
    (allProviders?.[row.provider]?.savedCredentials ?? []).some((saved) =>
      ids.includes(saved.id),
    );
  const hasRefusedAccount = onFile(row.rejectedCredentialIds);
  const awaitingReconnect = onFile(rejectedIds);

  const savedCredential = findSavedUserCredentialByProviderAndType(
    row.schema.credentials_provider ?? [],
    row.schema.credentials_types ?? [],
    row.schema.credentials_scopes,
    usableProviders,
    row.schema.discriminator_values,
  );

  async function grant(
    credential: Grantable,
    candidate?: SavedCredential,
  ): Promise<boolean> {
    if (!expertGrant) return false;
    setGrantError(null);
    // Resolve the account before the POST: validating afterwards tells the
    // user the grant failed while the expert already holds it.
    const account =
      candidate ??
      allProviders?.[row.provider]?.savedCredentials.find(
        (saved) => saved.id === credential.id,
      );
    if (!account || !grantableAmong(row, [account])) {
      setGrantError(UNUSABLE_ACCOUNT_ERROR);
      return false;
    }
    let response: Awaited<ReturnType<typeof grantCredentials>>;
    try {
      response = await grantCredentials({
        expertId: expertGrant.expertId,
        data: { credential_ids: [credential.id] },
      });
    } catch (error) {
      setGrantError(
        error instanceof ApiError ? error.message : GRANT_FAILED_ERROR,
      );
      return false;
    }
    if (!grantedCredentials.confirmGrant(credential.id, response)) {
      setGrantError(GRANT_FAILED_ERROR);
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
      if (!awaitingNewAccount || row.selected || !knownIds.current) return;
      const fresh = newlyConnectedCredential(
        row,
        allProviders,
        knownIds.current,
      );
      if (!fresh) return;
      setAwaitingNewAccount(false);
      setConnected(fresh.grantable);
      void grant(fresh.grantable, fresh.account);
      return;
    }
    if (awaitingNewAccount && knownIds.current) {
      const fresh = newlyConnectedCredential(
        row,
        allProviders,
        knownIds.current,
      );
      if (fresh) {
        setAwaitingNewAccount(false);
        setRenewedIds((ids) => [...ids, fresh.grantable.id]);
        return;
      }
    }
    // Cards stream in one commit at a time, so a row's schema can widen after
    // it auto-selected: a selection that no longer satisfies it must go, or
    // the row reads Connected while Proceed sends a credential missing the
    // scopes the later card asked for. `null` is the provider context's
    // "still loading" sentinel, where every lookup misses — clearing then
    // would drop a good selection on every mount. A refused credential never
    // fits, which is what turns a row holding one back into a Reconnect.
    if (allProviders && row.selected && !selectedStillFits) {
      row.select(undefined);
      return;
    }
    if (row.selected) return;
    // The account just signed in with wins over the others that also fit.
    // It is picked once the provider list has it, so it is judged on the
    // scopes the backend stored rather than on what the sign-in reported.
    const signedIn = pickable.find((c) => c.id === renewedIds.at(-1));
    if (signedIn) {
      row.select({
        id: signedIn.id,
        provider: row.provider,
        type: signedIn.type as CredentialsMetaInput["type"],
        title: signedIn.title,
      });
      return;
    }
    if (!savedCredential || hasRefusedAccount) return;
    row.select({
      id: savedCredential.id,
      provider: savedCredential.provider,
      type: savedCredential.type as CredentialsMetaInput["type"],
      title: savedCredential.title ?? undefined,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- row.select is rebuilt each render by the card
  }, [
    savedCredential?.id,
    row.selected,
    allProviders,
    awaitingNewAccount,
    expertGrant?.expertId,
    rejectedIds.join(),
    renewedIds.join(),
    hasRefusedAccount,
  ]);

  // Several saved accounts can satisfy one row. Nothing picks between them for
  // the user: the row offers them, and the backend runs on exactly that one.
  const pickable = expertGrant
    ? []
    : filterSystemCredentials(
        usableProviders?.[row.provider]?.savedCredentials ?? [],
      ).flatMap((saved) => grantableAmong(row, [saved]) ?? []);
  // The selected credential itself must still fit: that another account
  // does is no reason to keep this one and call it Connected.
  const selectedStillFits = pickable.some(
    (credential) => credential.id === row.selected?.id,
  );
  const hasChoice =
    !expertGrant &&
    !row.selected &&
    pickable.length > (hasRefusedAccount ? 0 : 1);
  // No saved account fits, and there are several that a fresh sign-in could
  // widen. Signing in without naming one requests only this card's scopes,
  // which the backend cannot merge into an account that holds others, so it
  // stored yet another credential beside them. The user names the account
  // instead; with exactly one, `upgradableCredentialID` already does. Only
  // where the row takes an OAuth credential at all: a card asking for an API
  // key runs no sign-in to aim, so naming an account there picks one the key
  // form then ignores.
  const updatable =
    expertGrant ||
    pickable.length > 0 ||
    !(row.schema.credentials_types ?? []).includes("oauth2")
      ? []
      : updatableAccounts(row, allProviders);

  async function pick(credential: Grantable): Promise<boolean> {
    row.select({
      id: credential.id,
      provider: row.provider,
      type: credential.type as CredentialsMetaInput["type"],
      title: credential.title,
    });
    row.onConnected();
    return true;
  }

  const grantableOptions = [
    ...(connected ? [connected] : []),
    ...(expertGrant?.credentials ?? []).filter((c) => c.id !== connected?.id),
  ];
  // A merged card whose own field is still empty leaves the row unanswered,
  // whatever the first target holds — saying "Granted" there claims a run can
  // proceed that cannot.
  const isSatisfied =
    !row.hasUnansweredTarget &&
    (expertGrant
      ? grantedCredentials.isSelectionGranted
      : !!row.selected && !rejectedIds.includes(row.selected.id));

  function openDialog() {
    knownIds.current = allProviders
      ? new Set(
          allProviders[row.provider]?.savedCredentials.map((c) => c.id) ?? [],
        )
      : null;
    setGrantError(null);
    setAwaitingNewAccount(false);
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
        ) : expertGrant && !isSatisfied ? (
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

      {isSatisfied ? (
        <span className="flex shrink-0 items-center gap-1.5 text-sm font-medium text-zinc-500">
          <Icon icon={CheckmarkCircle02Icon} size={16} />
          {expertGrant ? "Granted" : "Connected"}
        </span>
      ) : (
        <Button
          variant="primary"
          size="small"
          className="shrink-0"
          disabled={
            isGranting ||
            Boolean(
              expertGrant && (grantedCredentials.isPending || !allProviders),
            )
          }
          onClick={openDialog}
        >
          {buttonLabel({
            canGrant: grantableOptions.length > 0,
            awaitingReconnect,
            hasChoice,
          })}
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
            : hasChoice
              ? {
                  credentials: pickable,
                  onUse: pick,
                  isPending: false,
                  error: null,
                  purpose: "choose",
                }
              : updatable.length > 1
                ? {
                    credentials: updatable,
                    // The dialog runs the sign-in itself for this purpose.
                    onUse: async () => false,
                    isPending: false,
                    error: null,
                    purpose: "update",
                  }
                : undefined
        }
        open={isDialogOpen}
        onClose={() => setDialogOpen(false)}
        onConnected={(credential) => {
          if (!expertGrant) {
            // Without a reported credential nothing is known to be renewed,
            // so every refused account stays refused until the diff finds
            // the one this sign-in added.
            if (credential) setRenewedIds((ids) => [...ids, credential.id]);
            else setAwaitingNewAccount(true);
            row.onConnected(credential?.id);
            return;
          }
          // A flow that does not report its credential leaves the refresh to
          // find the new account.
          if (!credential) {
            setAwaitingNewAccount(true);
            return;
          }
          // Grant the reported credential directly: a re-auth that upgraded
          // an existing account keeps its id, which a refresh diff would
          // never surface. One that cannot satisfy the row is an error, not
          // a wait.
          const account = toSavedCredential(credential);
          const usable = grantableAmong(row, [account]);
          if (!usable) {
            setGrantError(UNUSABLE_ACCOUNT_ERROR);
            return;
          }
          setConnected(usable);
          void grant(usable, account);
        }}
      />
    </div>
  );
}

/** The credential among `candidates` that satisfies `row`, shaped for a
 *  grant. Matching reads nothing but the candidates, so a row whose provider
 *  has not loaded yet is still answerable. */
function grantableAmong(
  row: Row,
  candidates: SavedCredential[],
): Grantable | null {
  if (candidates.length === 0) return null;
  const match = findSavedUserCredentialByProviderAndType(
    row.schema.credentials_provider ?? [],
    row.schema.credentials_types ?? [],
    row.schema.credentials_scopes,
    {
      [row.provider]: {
        savedCredentials: candidates,
      } as CredentialsProviderData,
    },
    row.schema.discriminator_values,
  );
  return match
    ? { id: match.id, title: match.title ?? row.displayName, type: match.type }
    : null;
}

/** A refused account is named as such even when other accounts are offered
 *  beside it: the dialog lists them and its Add new signs in again. */
function buttonLabel({
  canGrant,
  awaitingReconnect,
  hasChoice,
}: {
  canGrant: boolean;
  awaitingReconnect: boolean;
  hasChoice: boolean;
}) {
  if (canGrant) return "Grant access";
  if (awaitingReconnect) return "Reconnect";
  if (hasChoice) return "Choose account";
  return "Connect";
}

/** `allProviders` without the credentials in `ids`. Keeps the `null` loading
 *  sentinel, and the list itself when there is nothing to drop. */
function withoutCredentials(
  allProviders: CredentialsProvidersContextType | null,
  ids: string[],
): CredentialsProvidersContextType | null {
  if (!allProviders || ids.length === 0) return allProviders;
  return Object.fromEntries(
    Object.entries(allProviders).map(([name, data]) => [
      name,
      data && {
        ...data,
        savedCredentials: data.savedCredentials.filter(
          (credential) => !ids.includes(credential.id),
        ),
      },
    ]),
  );
}

/** The API reports absent fields as null; the provider list omits them. */
function toSavedCredential(
  credential: CredentialsMetaResponse,
): SavedCredential {
  return {
    ...credential,
    title: credential.title ?? undefined,
    username: credential.username ?? undefined,
    scopes: credential.scopes ?? undefined,
    host: credential.host ?? undefined,
  };
}

/** The credential that satisfies `row` and did not exist before Connect was
 *  clicked, i.e. the account the user just signed in with, paired with the
 *  saved account it came from so granting it needs no second lookup — the
 *  provider list can have moved on again by then. */
function newlyConnectedCredential(
  row: Row,
  allProviders: CredentialsProvidersContextType | null,
  knownIds: Set<string>,
): { grantable: Grantable; account: SavedCredential } | null {
  const provider = allProviders?.[row.provider];
  if (!provider) return null;
  const added = provider.savedCredentials.filter((c) => !knownIds.has(c.id));
  const grantable = grantableAmong(row, added);
  if (!grantable) return null;
  const account = added.find((c) => c.id === grantable.id);
  return account ? { grantable, account } : null;
}

/** The user's own OAuth accounts for a provider: the ones a fresh sign-in can
 *  widen. API keys have nothing to re-authorise, and managed and system
 *  credentials are refused by the backend. The one-click upgrade and the
 *  account picker must agree on this set, or a provider counted one way and
 *  offered the other leaves the user with no path that upgrades in place. */
function updatableCredentials(
  provider: string,
  allProviders: CredentialsProvidersContextType | null,
) {
  return filterSystemCredentials(
    allProviders?.[provider]?.savedCredentials ?? [],
  ).filter(
    (credential) => credential.type === "oauth2" && !credential.is_managed,
  );
}

function updatableAccounts(
  row: Row,
  allProviders: CredentialsProvidersContextType | null,
): Grantable[] {
  return updatableCredentials(row.provider, allProviders).map((credential) => ({
    id: credential.id,
    title: credential.title ?? credential.username ?? row.displayName,
    type: credential.type,
  }));
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
  const oauthCredentials = updatableCredentials(provider, allProviders);
  return oauthCredentials.length === 1 ? oauthCredentials[0].id : undefined;
}

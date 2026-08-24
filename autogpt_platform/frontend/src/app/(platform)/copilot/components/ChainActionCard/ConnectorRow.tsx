"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ConnectCredentialDialog } from "@/components/contextual/CredentialsInput/components/ConnectCredentialDialog/ConnectCredentialDialog";
import { findSavedUserCredentialByProviderAndType } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";
import { ProviderAvatar } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/ProviderAvatar";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { useContext, useEffect, useState } from "react";
import type { ConnectorRow as Row } from "./helpers";

interface Props {
  row: Row;
}

export function ConnectorRow({ row }: Props) {
  const [isDialogOpen, setDialogOpen] = useState(false);
  const allProviders = useContext(CredentialsProvidersContext);

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

  useEffect(() => {
    if (row.selected || !savedCredential) return;
    row.select({
      id: savedCredential.id,
      provider: savedCredential.provider,
      type: savedCredential.type as CredentialsMetaInput["type"],
      title: savedCredential.title ?? undefined,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- row.select is rebuilt each render by the card
  }, [savedCredential?.id, row.selected]);

  return (
    <div className="flex items-center gap-3 px-4 py-3">
      <span className="flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-2xl border border-zinc-100 bg-white p-1.5">
        <ProviderAvatar id={row.provider} name={row.displayName} />
      </span>

      <div className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-sm font-medium text-zinc-900">
          {row.displayName}
        </span>
        {row.description && (
          <span className="truncate text-sm text-zinc-500">
            {row.description}
          </span>
        )}
      </div>

      {row.selected ? (
        <span className="flex shrink-0 items-center gap-1.5 text-sm font-medium text-zinc-500">
          <Icon icon={CheckmarkCircle02Icon} size={16} />
          Connected
        </span>
      ) : (
        <Button
          variant="primary"
          size="small"
          className="shrink-0"
          onClick={() => setDialogOpen(true)}
        >
          Connect
        </Button>
      )}

      <ConnectCredentialDialog
        schema={row.schema}
        provider={row.provider}
        displayName={row.displayName}
        open={isDialogOpen}
        onClose={() => setDialogOpen(false)}
      />
    </div>
  );
}

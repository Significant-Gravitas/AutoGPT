import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ExclamationTriangleIcon } from "@radix-ui/react-icons";
import { useState } from "react";
import { AyrshareConnectButton } from "../AyrshareConnectButton/AyrshareConnectButton";
import { CredentialRow } from "../CredentialRow/CredentialRow";
import { CredentialsSelect } from "../CredentialsSelect/CredentialsSelect";

type Credential = {
  id: string;
  title?: string;
  username?: string;
  type: string;
  provider: string;
  is_managed?: boolean;
};

function ProviderConnectRow({
  provider,
  displayName,
  actionButtonText,
  onAddCredential,
}: {
  provider: string;
  displayName: string;
  actionButtonText: string;
  onAddCredential: () => void;
}) {
  const src = `/integrations/${provider}.png`;
  const [broken, setBroken] = useState(false);

  return (
    <div className="flex h-14 w-full items-center gap-2.5 rounded-xl bg-neutral-100 px-3">
      {broken ? (
        <div
          aria-hidden
          className="flex size-7 shrink-0 items-center justify-center rounded-md bg-white text-[12px] font-semibold uppercase text-zinc-600"
        >
          {displayName.charAt(0)}
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt=""
          width={28}
          height={28}
          loading="lazy"
          className="size-7 shrink-0 object-contain"
          onError={() => setBroken(true)}
        />
      )}
      <span className="min-w-0 flex-1 truncate text-[14px] font-medium leading-[22px] text-zinc-800">
        {displayName}
      </span>
      <Button
        variant="primary"
        size="small"
        onClick={onAddCredential}
        className="shrink-0"
        type="button"
      >
        {actionButtonText}
      </Button>
    </div>
  );
}

type Props = {
  schema: BlockIOCredentialsSubSchema;
  provider: string;
  displayName: string;
  credentials: Credential[];
  selectedCredential?: CredentialsMetaInput;
  actionButtonText: string;
  isOptional: boolean;
  showTitle: boolean;
  readOnly: boolean;
  variant: "default" | "node";
  onSelectCredential: (credentialId: string) => void;
  onClearCredential: () => void;
  onAddCredential: () => void;
  onDeleteCredential?: (credential: { id: string; title: string }) => void;
};

export function CredentialsFlatView({
  schema,
  provider,
  displayName,
  credentials,
  selectedCredential,
  actionButtonText,
  isOptional,
  showTitle,
  readOnly,
  variant,
  onSelectCredential,
  onClearCredential,
  onAddCredential,
  onDeleteCredential,
}: Props) {
  const isNewToolUI = useGetFlag(Flag.NEW_TOOL_UI);
  const hasCredentials = credentials.length > 0;
  // Ayrshare has no user-settable credential — provisioning runs on the
  // server after the user clicks the Connect Social Media Accounts
  // button rendered below. Exposing "Add API key" / "Use a new API key"
  // here just confuses users into entering a random key.
  const isManagedOnlyProvider = provider === "ayrshare";
  const showAddAction = !readOnly && !isManagedOnlyProvider;
  const showAyrshareConnect = isManagedOnlyProvider && !readOnly;

  return (
    <>
      {showTitle && (
        <div className="mb-2 flex items-center gap-2">
          {isNewToolUI ? (
            <Text variant="small" className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1 text-zinc-600">
                {displayName} credentials
                {isOptional && (
                  <span className="font-normal text-gray-500">(optional)</span>
                )}
                {!isOptional && !selectedCredential && (
                  <span className="inline-flex items-center gap-1 text-red-600">
                    <ExclamationTriangleIcon className="size-3.5" />
                    <span className="font-normal">required</span>
                  </span>
                )}
              </span>
            </Text>
          ) : (
            <Text variant="large-medium" className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1">
                {displayName} credentials
                {isOptional && (
                  <span className="text-sm font-normal text-gray-500">
                    (optional)
                  </span>
                )}
                {!isOptional && !selectedCredential && (
                  <span className="inline-flex items-center gap-1 text-red-600">
                    <ExclamationTriangleIcon className="size-4" />
                    <span className="text-sm font-normal">required</span>
                  </span>
                )}
              </span>
            </Text>
          )}
          {schema.description && (
            <InformationTooltip description={schema.description} />
          )}
        </div>
      )}

      {hasCredentials ? (
        <>
          {(credentials.length > 1 || isOptional) && !readOnly ? (
            <CredentialsSelect
              credentials={credentials}
              provider={provider}
              displayName={displayName}
              selectedCredentials={selectedCredential}
              onSelectCredential={onSelectCredential}
              onClearCredential={onClearCredential}
              readOnly={readOnly}
              allowNone={isOptional}
              variant={variant}
            />
          ) : (
            <div className="mb-4 space-y-2">
              {credentials.map((credential) => (
                <CredentialRow
                  key={credential.id}
                  credential={credential}
                  provider={provider}
                  displayName={displayName}
                  onSelect={() => onSelectCredential(credential.id)}
                  onDelete={
                    onDeleteCredential && !credential.is_managed
                      ? () =>
                          onDeleteCredential({
                            id: credential.id,
                            title: credential.title || credential.id,
                          })
                      : undefined
                  }
                  readOnly={readOnly}
                />
              ))}
            </div>
          )}
          {showAddAction && (
            <Button
              variant="secondary"
              size="small"
              onClick={onAddCredential}
              className="w-fit"
              type="button"
            >
              {actionButtonText}
            </Button>
          )}
          {showAyrshareConnect && <AyrshareConnectButton className="mt-2" />}
        </>
      ) : showAddAction ? (
        isNewToolUI ? (
          <ProviderConnectRow
            provider={provider}
            displayName={displayName}
            actionButtonText={actionButtonText}
            onAddCredential={onAddCredential}
          />
        ) : (
          <Button
            variant="primary"
            size="small"
            onClick={onAddCredential}
            className="w-fit"
            type="button"
          >
            {actionButtonText}
          </Button>
        )
      ) : showAyrshareConnect ? (
        <AyrshareConnectButton className="mt-2" />
      ) : null}
    </>
  );
}

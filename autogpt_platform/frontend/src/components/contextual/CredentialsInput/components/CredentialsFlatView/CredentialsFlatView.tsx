import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { ExclamationTriangleIcon } from "@radix-ui/react-icons";
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
  const hasCredentials = credentials.length > 0;
  // Ayrshare has no user-settable credential — provisioning runs on the
  // server after the user clicks the explicit Connect Social Media
  // Accounts button rendered alongside the block.  Exposing "Add API
  // key" / "Use a new API key" here just confuses users into entering a
  // random key.
  const isManagedOnlyProvider = provider === "ayrshare";
  const showAddAction = !readOnly && !isManagedOnlyProvider;

  return (
    <>
      {showTitle && (
        <div className="mb-2 flex items-center gap-2">
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
        </>
      ) : showAddAction ? (
        <Button
          variant="primary"
          size="small"
          onClick={onAddCredential}
          className="w-fit"
          type="button"
        >
          {actionButtonText}
        </Button>
      ) : (
        isManagedOnlyProvider &&
        !readOnly && (
          <Text variant="body" className="text-zinc-500">
            Click <strong>Connect Social Media Accounts</strong> above to set up
            your managed {displayName} profile.
          </Text>
        )
      )}
    </>
  );
}

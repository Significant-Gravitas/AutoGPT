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
}: Props) {
  const hasCredentials = credentials.length > 0;

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
                  readOnly={readOnly}
                />
              ))}
            </div>
          )}
          {!readOnly && (
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
      ) : (
        !readOnly && (
          <Button
            variant="secondary"
            size="small"
            onClick={onAddCredential}
            className="w-fit"
            type="button"
          >
            {actionButtonText}
          </Button>
        )
      )}
    </>
  );
}

import React from "react";
import { FieldProps } from "@rjsf/utils";
import { useCredentialField } from "./useCredentialField";
import { SelectCredential } from "./SelectCredential";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { APIKeyCredentialsModal } from "./models/APIKeyCredentialModal/APIKeyCredentialModal";
import { OAuthCredentialModal } from "./models/OAuthCredentialModal/OAuthCredentialModal";
import { PasswordCredentialsModal } from "./models/PasswordCredentialModal/PasswordCredentialModal";
import { HostScopedCredentialsModal } from "./models/HostScopedCredentialsModal/HostScopedCredentialsModal";
import { Switch } from "@/components/atoms/Switch/Switch";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useShallow } from "zustand/react/shallow";

export const CredentialsField = (props: FieldProps) => {
  const {
    formData = {},
    onChange,
    required: _required,
    schema,
    formContext,
  } = props;

  const nodeId = formContext.nodeId;
  // Only show the optional toggle when editing blocks in the builder canvas
  const showOptionalToggle = formContext.showOptionalToggle !== false && nodeId;

  const { credentialsOptional, setCredentialsOptional } = useNodeStore(
    useShallow((state) => ({
      credentialsOptional: state.getCredentialsOptional(nodeId),
      setCredentialsOptional: state.setCredentialsOptional,
    })),
  );

  const {
    credentials,
    isCredentialListLoading,
    supportsApiKey,
    supportsOAuth2,
    supportsUserPassword,
    supportsHostScoped,
    credentialsExists,
    credentialProvider,
    setCredential,
    discriminatorValue,
  } = useCredentialField({
    credentialSchema: schema as BlockIOCredentialsSubSchema,
    formData,
    nodeId,
    onChange,
    disableAutoSelect: credentialsOptional,
  });

  if (isCredentialListLoading) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-8 w-full rounded-xlarge" />
        <Skeleton className="h-8 w-[30%] rounded-xlarge" />
      </div>
    );
  }

  if (!credentialProvider) {
    return null;
  }

  return (
    <div className="flex flex-col gap-2">
      {credentialsExists && (
        <SelectCredential
          credentials={credentials}
          value={formData.id || ""}
          onChange={setCredential}
          disabled={false}
          label="Credential"
          placeholder={
            credentialsOptional
              ? "Select credential (optional)"
              : "Select credential"
          }
        />
      )}

      <div className="flex flex-wrap gap-2">
        {supportsApiKey && (
          <APIKeyCredentialsModal
            schema={schema as BlockIOCredentialsSubSchema}
            provider={credentialProvider}
          />
        )}
        {supportsOAuth2 && (
          <OAuthCredentialModal provider={credentialProvider} />
        )}
        {supportsUserPassword && (
          <PasswordCredentialsModal provider={credentialProvider} />
        )}
        {supportsHostScoped && discriminatorValue && (
          <HostScopedCredentialsModal
            schema={schema as BlockIOCredentialsSubSchema}
            provider={credentialProvider}
            discriminatorValue={discriminatorValue}
          />
        )}
      </div>

      {/* Optional credentials toggle - only show in builder canvas, not run dialogs */}
      {showOptionalToggle && (
        <div className="mt-1 flex items-center gap-2">
          <Switch
            id={`credentials-optional-${nodeId}`}
            checked={credentialsOptional}
            onCheckedChange={(checked) =>
              setCredentialsOptional(nodeId, checked)
            }
          />
          <label
            htmlFor={`credentials-optional-${nodeId}`}
            className="cursor-pointer text-xs text-gray-500"
          >
            Optional - skip block if not configured
          </label>
        </div>
      )}
    </div>
  );
};

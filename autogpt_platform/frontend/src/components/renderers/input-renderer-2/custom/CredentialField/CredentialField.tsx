import React from "react";
import { FieldProps, getTemplate, getUiOptions } from "@rjsf/utils";
import { useCredentialField } from "./useCredentialField";
import { SelectCredential } from "./components/SelectCredential";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { APIKeyCredentialsModal } from "./components/models/APIKeyCredentialModal/APIKeyCredentialModal";
import { OAuthCredentialModal } from "./components/models/OAuthCredentialModal/OAuthCredentialModal";
import { PasswordCredentialsModal } from "./components/models/PasswordCredentialModal/PasswordCredentialModal";
import { HostScopedCredentialsModal } from "./components/models/HostScopedCredentialsModal/HostScopedCredentialsModal";
import { CredentialFieldTitle } from "./components/CredentialFieldTitle";

export const CredentialsField = (props: FieldProps) => {
  const {
    formData = {},
    onChange,
    required: _required,
    schema,
    registry,
    fieldPathId,
  } = props;

  const formContext = registry.formContext;
  const uiOptions = getUiOptions(props.uiSchema);

  const handleChange = (newValue: any) => {
    onChange(newValue, fieldPathId.path);
  };

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
    nodeId: formContext.nodeId,
    onChange: handleChange,
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
      <CredentialFieldTitle
        fieldPathId={fieldPathId}
        registry={registry}
        uiOptions={uiOptions}
        schema={schema}
      />
      {credentialsExists && (
        <SelectCredential
          credentials={credentials}
          value={formData.id || ""}
          onChange={setCredential}
          disabled={false}
          label="Credential"
          placeholder="Select credential"
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
    </div>
  );
};

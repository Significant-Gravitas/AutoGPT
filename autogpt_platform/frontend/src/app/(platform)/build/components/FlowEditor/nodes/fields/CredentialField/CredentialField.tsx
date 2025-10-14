import React, { useEffect } from "react";
import { FieldProps } from "@rjsf/utils";
import { useCredentialField } from "./useCredentialField";
import { SelectCredential } from "./SelectCredential";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { APIKeyCredentialsModal } from "./models/APIKeyCredentialModal/APIKeyCredentialModal";
import { OAuthCredentialModal } from "./models/OAuthCredentialModal/OAuthCredentialModal";
import { PasswordCredentialsModal } from "./models/PasswordCredentialModal/PasswordCredentialModal";

export const CredentialsField = (props: FieldProps) => {
  const {
    formData = {},
    onChange,
    required: _required,
    schema,
    formContext,
  } = props;
  const {
    credentials,
    isCredentialListLoading,
    supportsApiKey,
    supportsOAuth2,
    supportsUserPassword,
    credentialsExists,
    credentialProvider,
  } = useCredentialField({
    credentialSchema: schema as BlockIOCredentialsSubSchema,
    nodeId: formContext.nodeId,
  });

  const setField = (key: string, value: any) =>
    onChange({ ...formData, [key]: value });

  // This is to set the latest credential as the default one [currently, latest means last one in the list of credentials]
  useEffect(() => {
    if (!isCredentialListLoading && credentials.length > 0 && !formData.id) {
      const latestCredential = credentials[credentials.length - 1];
      setField("id", latestCredential.id);
    }
  }, [isCredentialListLoading, credentials, formData.id]);

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
          value={formData.id}
          onChange={(value) => setField("id", value)}
          disabled={false}
          label="Credential"
          placeholder="Select credential"
        />
      )}

      <div>
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
      </div>
    </div>
  );
};

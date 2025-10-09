import React, { useEffect } from "react";
import { FieldProps } from "@rjsf/utils";
import { useCredentialField } from "./useCredentialField";
import { SelectCredential } from "./SelectCredential";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { APIKeyCredentialsModal } from "./models/APIKeyCredentialModal/APIKeyCredentialModal";
import { OAuthCredentialModal } from "./models/OAuthCredentialModal/OAuthCredentialModal";

export const CredentialsField = (props: FieldProps) => {
  const { formData = {}, onChange, required: _required, schema } = props;
  const {
    credentials,
    isCredentialListLoading,
    supportsApiKey,
    supportsOAuth2,
    credentialsExists,
  } = useCredentialField({
    credentialSchema: schema as BlockIOCredentialsSubSchema,
  });

  const setField = (key: string, value: any) =>
    onChange({ ...formData, [key]: value });

  useEffect(() => {
    if (!isCredentialListLoading && credentials.length > 0 && !formData.id) {
      const latestCredential = credentials[credentials.length - 1];
      setField("id", latestCredential.id);
    }
  }, [isCredentialListLoading, credentials, formData.id]);

  const handleCredentialCreated = (credentialId: string) => {
    setField("id", credentialId);
  };

  if (isCredentialListLoading) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-8 w-full rounded-xlarge" />
        <Skeleton className="h-8 w-[30%] rounded-xlarge" />
      </div>
    );
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
            onSuccess={handleCredentialCreated}
          />
        )}
        {supportsOAuth2 && (
          <OAuthCredentialModal provider={schema.credentials_provider[0]} />
        )}
      </div>
    </div>
  );
};

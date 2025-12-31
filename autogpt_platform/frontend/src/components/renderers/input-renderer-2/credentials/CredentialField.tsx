import React from "react";
import {
  FieldProps,
  FormContextType,
  StrictRJSFSchema,
  RJSFSchema,
} from "@rjsf/utils";
import { useCredentialField } from "./useCredentialField";
import { SelectCredential } from "./SelectCredential";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { APIKeyCredentialsModal } from "./models/APIKeyCredentialModal/APIKeyCredentialModal";
import { OAuthCredentialModal } from "./models/OAuthCredentialModal/OAuthCredentialModal";
import { PasswordCredentialsModal } from "./models/PasswordCredentialModal/PasswordCredentialModal";
import { HostScopedCredentialsModal } from "./models/HostScopedCredentialsModal/HostScopedCredentialsModal";

export const CredentialsField = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(
  props: FieldProps<T, S, F>,
) => {
  const {
    formData = {},
    onChange,
    required: _required,
    schema,
    registry,
  } = props;
  const formContext = registry.formContext;
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
    credentialSchema: schema as unknown as BlockIOCredentialsSubSchema,
    formData: formData as Record<string, any>,
    nodeId: formContext.nodeId,
    onChange,
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
          value={((formData as Record<string, any>).id as string) || ""}
          onChange={setCredential}
          disabled={false}
          label="Credential"
          placeholder="Select credential"
        />
      )}

      <div className="flex flex-wrap gap-2">
        {supportsApiKey && (
          <APIKeyCredentialsModal
            schema={schema as unknown as BlockIOCredentialsSubSchema}
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
            schema={schema as unknown as BlockIOCredentialsSubSchema}
            provider={credentialProvider}
            discriminatorValue={discriminatorValue}
          />
        )}
      </div>
    </div>
  );
};

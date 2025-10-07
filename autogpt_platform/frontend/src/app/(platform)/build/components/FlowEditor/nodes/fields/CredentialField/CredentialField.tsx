import React, { useEffect } from "react";
import { FieldProps } from "@rjsf/utils";
import { useCredentialField } from "./useCredentialField";
import { KeyIcon, PlusIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { SelectCredential } from "./SelectCredential";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { APIKeyCredentialsModal } from "./models/APIKeyCredentialModal/APIKeyCredentialModal";
import { Text } from "@/components/atoms/Text/Text";

export const CredentialsField = (props: FieldProps) => {
  const { formData = {}, onChange, required: _required, schema } = props;
  const {
    credentials,
    isCredentialListLoading,
    supportsApiKey,
    supportsOAuth2,
    isAPIKeyModalOpen,
    setIsAPIKeyModalOpen,
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
          <>
            <APIKeyCredentialsModal
              schema={schema as BlockIOCredentialsSubSchema}
              open={isAPIKeyModalOpen}
              onClose={() => setIsAPIKeyModalOpen(false)}
              onSuccess={handleCredentialCreated}
            />
            <Button
              type="button"
              className="w-auto min-w-0"
              size="small"
              onClick={() => setIsAPIKeyModalOpen(true)}
            >
              <KeyIcon />
              <Text variant="body-medium" className="!text-white opacity-100">
                Add API key
              </Text>
            </Button>
          </>
        )}
        {supportsOAuth2 && (
          <Button type="button" className="w-fit" size="small">
            <PlusIcon />
            Add OAuth2
          </Button>
        )}
      </div>
    </div>
  );
};

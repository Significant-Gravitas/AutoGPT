import React from "react";
import { FieldProps } from "@rjsf/utils";
import { useCredentialField } from "./useCredentialField";
import { filterCredentialsByProvider } from "./helpers";
import { PlusIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { SelectCredential } from "./SelectCredential";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";

export const CredentialsField = (props: FieldProps) => {
  const { formData = {}, onChange, required: _required, schema } = props;
  const { credentials, isCredentialListLoading } = useCredentialField();

  const credentialProviders = schema.credentials_provider;
  const { credentials: filteredCredentials, exists: credentialsExists } =
    filterCredentialsByProvider(credentials, credentialProviders);

  const setField = (key: string, value: any) =>
    onChange({ ...formData, [key]: value });

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
          credentials={filteredCredentials}
          value={formData.id}
          onChange={(value) => setField("id", value)}
          disabled={false}
          label="Credential"
          placeholder="Select credential"
        />
      )}

      {/* TODO :  We need to add a modal to add a new credential */}
      <Button type="button" className="w-fit" size="small">
        <PlusIcon /> Add API Key
      </Button>
    </div>
  );
};

import React from "react";
import { FieldProps } from "@rjsf/utils";
import { Input } from "@/components/atoms/Input/Input";
import { useCredentialField } from "./useCredentialField";
import { filterCredentialsByProvider } from "./helpers";
import { CaretDownIcon, CheckIcon, KeyIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { SelectCredential } from "./SelectCredential";

// We need to add all the logic for the credential fields here
export const CredentialsField = (props: FieldProps) => {
  const { formData = {}, onChange, required: _required, schema } = props;
  const { credentials, isCredentialListLoading } = useCredentialField();

  const credentialProvider = schema.credentials_provider;
  const { credentials: filteredCredentials, exists: credentialsExists } =
    filterCredentialsByProvider(credentials, credentialProvider);

  const _credentialType = schema.credentials_types;
  const _description = schema.description;
  const _title = schema.title;

  console.log("schema", schema);

  // Helper to update one property
  const setField = (key: string, value: any) =>
    onChange({ ...formData, [key]: value });

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
        <KeyIcon /> Add API Key
      </Button>
    </div>
  );
};

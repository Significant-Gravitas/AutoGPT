import React from "react";
import { FieldProps } from "@rjsf/utils";
import { Input } from "@/components/atoms/Input/Input";

// We need to add all the logic for the credential fields here
export const CredentialsField = (props: FieldProps) => {
  const { formData = {}, onChange, required: _required, schema } = props;

  const _credentialProvider = schema.credentials_provider;
  const _credentialType = schema.credentials_types;
  const _description = schema.description;
  const _title = schema.title;

  // Helper to update one property
  const setField = (key: string, value: any) =>
    onChange({ ...formData, [key]: value });

  return (
    <div className="flex flex-col gap-2">
      <Input
        hideLabel={true}
        label={""}
        id="credentials-id"
        type="text"
        value={formData.id || ""}
        onChange={(e) => setField("id", e.target.value)}
        placeholder="Enter your API Key"
        required
        size="small"
        wrapperClassName="mb-0"
      />
    </div>
  );
};

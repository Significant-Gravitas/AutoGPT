import React from "react";
import { FieldProps } from "@rjsf/utils";
import { Input } from "@/components/atoms/Input/Input";

// We need to add all the logic for the credential fields here
export const CredentialsField = (props: FieldProps) => {
  const { formData = {}, onChange, required, schema } = props;

  const credentialProvider = schema.credentials_provider;
  const credentialType = schema.credentials_types;
  const description = schema.description;
  const title = schema.title;

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

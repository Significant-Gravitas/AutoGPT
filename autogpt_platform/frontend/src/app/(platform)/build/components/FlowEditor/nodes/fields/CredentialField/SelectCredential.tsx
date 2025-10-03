import React from "react";
import { Select } from "@/components/atoms/Select/Select";
import { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { KeyIcon } from "@phosphor-icons/react";

type SelectCredentialProps = {
  credentials: CredentialsMetaResponse[];
  value?: string;
  onChange: (credentialId: string) => void;
  disabled?: boolean;
  label?: string;
  placeholder?: string;
};

export const SelectCredential: React.FC<SelectCredentialProps> = ({
  credentials,
  value,
  onChange,
  disabled = false,
  label = "Credential",
  placeholder = "Select credential",
}) => {
  const options = credentials.map((cred) => {
    let details: string[] = [];
    if (cred.title && cred.title !== cred.provider) {
      details.push(cred.title);
    }
    if (cred.username) {
      details.push(cred.username);
    }
    if (cred.host) {
      details.push(cred.host);
    }
    const label =
      details.length > 0
        ? `${cred.provider} (${details.join(" - ")})`
        : cred.provider;

    return {
      value: cred.id,
      label,
      icon: <KeyIcon className="h-4 w-4" />,
    };
  });

  return (
    <Select
      label={label}
      id="select-credential"
      wrapperClassName="!mb-0"
      value={value}
      onValueChange={onChange}
      options={options}
      disabled={disabled}
      placeholder={placeholder}
      size="small"
      hideLabel
    />
  );
};

import React from "react";
import { Select } from "@/components/atoms/Select/Select";
import { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import {
  ArrowSquareOutIcon,
  KeyholeIcon,
  KeyIcon,
} from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import Link from "next/link";
import { providerIcons } from "./helpers";

type SelectCredentialProps = {
  credentials: CredentialsMetaResponse[];
  value?: string;
  defaultValue?: string;
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
    const details: string[] = [];
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

    const Icon = providerIcons[cred.provider];
    const icon =
      cred.type === "oauth2" ? (
        Icon ? (
          <Icon />
        ) : (
          <KeyholeIcon />
        )
      ) : (
        <KeyIcon className="h-4 w-4" />
      );

    return {
      value: cred.id,
      label,
      icon,
    };
  });

  return (
    <div className="flex w-full items-center gap-2">
      <Select
        label={label}
        id="select-credential"
        wrapperClassName="!mb-0 flex-1 !max-w-[90%]"
        value={value}
        onValueChange={onChange}
        options={options}
        disabled={disabled}
        placeholder={placeholder}
        size="small"
        hideLabel
      />
      <Link href={`/profile/integrations`}>
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8 border-zinc-300 p-0"
        >
          <ArrowSquareOutIcon className="h-4 w-4 text-zinc-600" />
        </Button>
      </Link>
    </div>
  );
};

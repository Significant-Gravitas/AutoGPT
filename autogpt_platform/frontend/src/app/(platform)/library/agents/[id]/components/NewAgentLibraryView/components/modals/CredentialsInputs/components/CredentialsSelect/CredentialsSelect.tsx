import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";
import { getCredentialDisplayName } from "../../helpers";
import { CredentialRow } from "../CredentialRow/CredentialRow";

interface Props {
  credentials: Array<{
    id: string;
    title?: string;
    username?: string;
    type: string;
    provider: string;
  }>;
  provider: string;
  displayName: string;
  selectedCredentials?: CredentialsMetaInput;
  onSelectCredential: (credentialId: string) => void;
  onClearCredential?: () => void;
  readOnly?: boolean;
  allowNone?: boolean;
  /** When "node", applies compact styling for node context */
  variant?: "default" | "node";
}

export function CredentialsSelect({
  credentials,
  provider,
  displayName,
  selectedCredentials,
  onSelectCredential,
  onClearCredential,
  readOnly = false,
  allowNone = true,
  variant = "default",
}: Props) {
  function handleValueChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const value = e.target.value;
    if (value === "__none__") {
      onClearCredential?.();
    } else {
      onSelectCredential(value);
    }
  }

  const selectedCredential = selectedCredentials
    ? credentials.find((c) => c.id === selectedCredentials.id)
    : null;

  const displayCredential = selectedCredential
    ? {
        id: selectedCredential.id,
        title: selectedCredential.title,
        username: selectedCredential.username,
        type: selectedCredential.type,
        provider: selectedCredential.provider,
      }
    : allowNone
      ? {
          id: "__none__",
          title: "None (skip this credential)",
          type: "none",
          provider: provider,
        }
      : {
          id: "__placeholder__",
          title: "Select credential",
          type: "placeholder",
          provider: provider,
        };

  return (
    <div className="mb-4 w-full">
      <div className="relative">
        <select
          value={selectedCredentials?.id ?? "__none__"}
          onChange={handleValueChange}
          disabled={readOnly}
          className="absolute inset-0 z-10 cursor-pointer opacity-0"
          aria-label={`Select ${displayName} credential`}
        >
          {allowNone ? (
            <option value="__none__">None (skip this credential)</option>
          ) : (
            <option value="__none__" disabled hidden>
              Select a credential
            </option>
          )}
          {credentials.map((credential) => (
            <option key={credential.id} value={credential.id}>
              {getCredentialDisplayName(credential, displayName)}
            </option>
          ))}
        </select>
        <div className="rounded-medium border border-zinc-200 bg-white">
          <CredentialRow
            credential={displayCredential}
            provider={provider}
            displayName={displayName}
            onSelect={() => {}}
            onDelete={() => {}}
            readOnly={readOnly}
            asSelectTrigger={true}
            variant={variant}
          />
        </div>
      </div>
    </div>
  );
}

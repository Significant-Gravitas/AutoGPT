import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";
import { useEffect, useRef } from "react";
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

  // Resolve the selected credential — treat stale/deleted IDs as unselected
  const selectedCredential = selectedCredentials
    ? credentials.find((c) => c.id === selectedCredentials.id)
    : null;

  // When credentials exist and nothing is matched,
  // default to the first credential instead of "None"
  const effectiveCredential =
    selectedCredential ?? (credentials.length > 0 ? credentials[0] : null);

  const displayCredential = effectiveCredential
    ? {
        id: effectiveCredential.id,
        title: effectiveCredential.title,
        username: effectiveCredential.username,
        type: effectiveCredential.type,
        provider: effectiveCredential.provider,
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

  // Use matched credential ID (not the raw selectedCredentials.id which may be stale)
  const defaultValue =
    effectiveCredential?.id ??
    (credentials.length > 0 ? credentials[0].id : "__none__");

  // Notify parent when defaulting to a credential so the value is captured on submit
  const hasNotifiedDefault = useRef(false);
  useEffect(() => {
    if (hasNotifiedDefault.current) return;
    if (selectedCredential) return; // Already matched — no need to override
    if (credentials.length > 0) {
      hasNotifiedDefault.current = true;
      onSelectCredential(credentials[0].id);
    }
  }, [credentials, selectedCredential, onSelectCredential]);

  return (
    <div className="mb-4 w-full">
      <div className="relative">
        <select
          value={defaultValue}
          onChange={handleValueChange}
          disabled={readOnly}
          className="absolute inset-0 z-10 cursor-pointer opacity-0"
          aria-label={`Select ${displayName} credential`}
        >
          {credentials.map((credential) => (
            <option key={credential.id} value={credential.id}>
              {getCredentialDisplayName(credential, displayName)}
            </option>
          ))}
          {allowNone ? (
            <option value="__none__">None (skip this credential)</option>
          ) : (
            <option value="__none__" disabled hidden>
              Select a credential
            </option>
          )}
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

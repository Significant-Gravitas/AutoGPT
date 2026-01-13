import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { useEffect } from "react";
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
  // Auto-select first credential if none is selected (only if allowNone is false)
  useEffect(() => {
    if (!allowNone && !selectedCredentials && credentials.length > 0) {
      onSelectCredential(credentials[0].id);
    }
  }, [allowNone, selectedCredentials, credentials, onSelectCredential]);

  const handleValueChange = (value: string) => {
    if (value === "__none__") {
      onClearCredential?.();
    } else {
      onSelectCredential(value);
    }
  };

  return (
    <div className="mb-4 w-full">
      <Select
        value={selectedCredentials?.id || (allowNone ? "__none__" : "")}
        onValueChange={handleValueChange}
      >
        <SelectTrigger
          className={cn(
            "h-auto min-h-12 w-full rounded-medium border-zinc-200 p-0 pr-4 shadow-none",
            variant === "node" && "overflow-hidden",
          )}
        >
          {selectedCredentials ? (
            <SelectValue key={selectedCredentials.id} asChild>
              <CredentialRow
                credential={{
                  id: selectedCredentials.id,
                  title: selectedCredentials.title || undefined,
                  type: selectedCredentials.type,
                  provider: selectedCredentials.provider,
                }}
                provider={provider}
                displayName={displayName}
                onSelect={() => {}}
                onDelete={() => {}}
                readOnly={readOnly}
                asSelectTrigger={true}
                variant={variant}
              />
            </SelectValue>
          ) : (
            <SelectValue key="placeholder" placeholder="Select credential" />
          )}
        </SelectTrigger>
        <SelectContent>
          {allowNone && (
            <SelectItem key="__none__" value="__none__">
              <div className="flex items-center gap-2">
                <Text variant="body" className="tracking-tight text-gray-500">
                  None (skip this credential)
                </Text>
              </div>
            </SelectItem>
          )}
          {credentials.map((credential) => (
            <SelectItem key={credential.id} value={credential.id}>
              <div className="flex items-center gap-2">
                <Text variant="body" className="tracking-tight">
                  {getCredentialDisplayName(credential, displayName)}
                </Text>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

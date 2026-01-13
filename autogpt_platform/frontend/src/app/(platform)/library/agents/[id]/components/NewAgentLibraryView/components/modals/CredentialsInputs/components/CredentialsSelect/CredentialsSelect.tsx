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
            "h-auto min-h-12 w-full rounded-medium p-0 pr-4 shadow-none",
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
          ) : allowNone ? (
            <SelectValue key="__none__" asChild>
              <div
                className={cn(
                  "flex items-center gap-3 rounded-medium border border-zinc-200 bg-white p-3 transition-colors",
                  variant === "node"
                    ? "min-w-0 flex-1 overflow-hidden border-0 bg-transparent"
                    : "border-0 bg-transparent",
                )}
              >
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-zinc-200">
                  <Text
                    variant="body"
                    className="text-xs font-medium text-zinc-500"
                  >
                    —
                  </Text>
                </div>
                <div
                  className={cn(
                    "flex min-w-0 flex-1 flex-nowrap items-center gap-4",
                    variant === "node" && "overflow-hidden",
                  )}
                >
                  <Text
                    variant="body"
                    className={cn("tracking-tight text-zinc-500")}
                  >
                    None (skip this credential)
                  </Text>
                </div>
              </div>
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

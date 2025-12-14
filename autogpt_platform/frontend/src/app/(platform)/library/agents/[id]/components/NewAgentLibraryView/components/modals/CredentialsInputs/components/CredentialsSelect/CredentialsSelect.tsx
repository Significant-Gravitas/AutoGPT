import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
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
  readOnly?: boolean;
}

export function CredentialsSelect({
  credentials,
  provider,
  displayName,
  selectedCredentials,
  onSelectCredential,
  readOnly = false,
}: Props) {
  // Auto-select first credential if none is selected
  useEffect(() => {
    if (!selectedCredentials && credentials.length > 0) {
      onSelectCredential(credentials[0].id);
    }
  }, [selectedCredentials, credentials, onSelectCredential]);

  return (
    <div className="mb-4 w-full">
      <Select
        value={selectedCredentials?.id || ""}
        onValueChange={(value) => onSelectCredential(value)}
      >
        <SelectTrigger className="h-auto min-h-12 w-full rounded-medium border-zinc-200 p-0 pr-4 shadow-none">
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
              />
            </SelectValue>
          ) : (
            <SelectValue key="placeholder" placeholder="Select credential" />
          )}
        </SelectTrigger>
        <SelectContent>
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

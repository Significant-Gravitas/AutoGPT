import { useState, useMemo } from "react";
import type { CredentialInfo } from "./ChatCredentialsSetup";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api";

export function useChatCredentialsSetup(credentials: CredentialInfo[]) {
  const [selectedCredentials, setSelectedCredentials] = useState<
    Record<string, CredentialsMetaInput>
  >({});

  // Check if all credentials are configured
  const isAllComplete = useMemo(
    function checkAllComplete() {
      if (credentials.length === 0) return false;
      return credentials.every((cred) => selectedCredentials[cred.provider]);
    },
    [credentials, selectedCredentials],
  );

  function handleCredentialSelect(
    provider: string,
    credential?: CredentialsMetaInput,
  ) {
    if (credential) {
      setSelectedCredentials((prev) => ({
        ...prev,
        [provider]: credential,
      }));
    }
  }

  return {
    selectedCredentials,
    isAllComplete,
    handleCredentialSelect,
  };
}

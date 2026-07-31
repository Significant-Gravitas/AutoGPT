import type { CredentialsMetaResponse } from "@/lib/autogpt-server-api";
import type { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";

export function getSavedCodexCredentials(
  providers: CredentialsProvidersContextType | null,
): CredentialsMetaResponse[] {
  return (
    providers?.codex?.savedCredentials.filter(
      (credential) =>
        credential.provider === "codex" && credential.type === "oauth2",
    ) ?? []
  );
}

export function hasSavedCodexCredential(
  providers: CredentialsProvidersContextType | null,
  credentialId: string,
): boolean {
  return getSavedCodexCredentials(providers).some(
    (credential) => credential.id === credentialId,
  );
}

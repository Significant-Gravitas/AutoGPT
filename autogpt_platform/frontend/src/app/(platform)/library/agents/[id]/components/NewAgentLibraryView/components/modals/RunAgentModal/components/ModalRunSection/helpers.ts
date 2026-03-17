import { CredentialsMetaResponseType } from "@/app/api/__generated__/models/credentialsMetaResponseType";

export function getCredentialTypeDisplayName(type: string): string {
  const typeDisplayMap: Record<CredentialsMetaResponseType, string> = {
    [CredentialsMetaResponseType.api_key]: "API key",
    [CredentialsMetaResponseType.oauth2]: "OAuth2",
    [CredentialsMetaResponseType.user_password]: "Username/Password",
    [CredentialsMetaResponseType.host_scoped]: "Host-Scoped",
  };

  return typeDisplayMap[type as CredentialsMetaResponseType] || type;
}

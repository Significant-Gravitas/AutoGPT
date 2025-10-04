import { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";

export const filterCredentialsByProvider = (
  credentials: CredentialsMetaResponse[] | undefined,
  provider: string[],
) => {
  const filtered =
    credentials?.filter((credential) =>
      provider.includes(credential.provider),
    ) ?? [];
  return {
    credentials: filtered,
    exists: filtered.length > 0,
  };
};

export function toDisplayName(provider: string): string {
  console.log("provider", provider);
  // Special cases that need manual handling
  const specialCases: Record<string, string> = {
    aiml_api: "AI/ML",
    d_id: "D-ID",
    e2b: "E2B",
    llama_api: "Llama API",
    open_router: "Open Router",
    smtp: "SMTP",
    revid: "Rev.ID",
  };

  if (specialCases[provider]) {
    return specialCases[provider];
  }

  // General case: convert snake_case to Title Case
  return provider
    .split(/[_-]/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
}

export function isCredentialFieldSchema(schema: any): boolean {
  return (
    typeof schema === "object" &&
    schema !== null &&
    "credentials_provider" in schema
  );
}

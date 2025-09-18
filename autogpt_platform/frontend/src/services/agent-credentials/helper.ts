// --8<-- [start:CredentialsProviderNames]
// Helper function to convert provider names to display names
export function toDisplayName(provider: string): string {
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

// Provider display names are now generated dynamically by toDisplayName function
// --8<-- [end:CredentialsProviderNames]

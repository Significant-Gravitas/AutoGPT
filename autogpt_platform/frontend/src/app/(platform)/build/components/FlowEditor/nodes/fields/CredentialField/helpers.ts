import { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
// Need to replace these icons with phosphor icons
import {
  FaDiscord,
  FaMedium,
  FaGithub,
  FaGoogle,
  FaHubspot,
  FaTwitter,
} from "react-icons/fa";
import { GoogleLogoIcon, KeyIcon, NotionLogoIcon } from "@phosphor-icons/react";

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

export const providerIcons: Partial<
  Record<string, React.FC<{ className?: string }>>
> = {
  aiml_api: KeyIcon,
  anthropic: KeyIcon,
  apollo: KeyIcon,
  e2b: KeyIcon,
  github: FaGithub,
  google: GoogleLogoIcon,
  groq: KeyIcon,
  http: KeyIcon,
  notion: NotionLogoIcon,
  nvidia: KeyIcon,
  discord: FaDiscord,
  d_id: KeyIcon,
  google_maps: FaGoogle,
  jina: KeyIcon,
  ideogram: KeyIcon,
  linear: KeyIcon,
  medium: FaMedium,
  mem0: KeyIcon,
  ollama: KeyIcon,
  openai: KeyIcon,
  openweathermap: KeyIcon,
  open_router: KeyIcon,
  llama_api: KeyIcon,
  pinecone: KeyIcon,
  enrichlayer: KeyIcon,
  slant3d: KeyIcon,
  screenshotone: KeyIcon,
  smtp: KeyIcon,
  replicate: KeyIcon,
  reddit: KeyIcon,
  fal: KeyIcon,
  revid: KeyIcon,
  twitter: FaTwitter,
  unreal_speech: KeyIcon,
  exa: KeyIcon,
  hubspot: FaHubspot,
  smartlead: KeyIcon,
  todoist: KeyIcon,
  zerobounce: KeyIcon,
};

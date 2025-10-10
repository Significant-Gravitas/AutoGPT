import { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import {
  GoogleLogoIcon,
  KeyholeIcon,
  NotionLogoIcon,
  DiscordLogoIcon,
  MediumLogoIcon,
  GithubLogoIcon,
  TwitterLogoIcon,
  Icon,
} from "@phosphor-icons/react";

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

export const providerIcons: Partial<Record<string, Icon>> = {
  aiml_api: KeyholeIcon,
  anthropic: KeyholeIcon,
  apollo: KeyholeIcon,
  e2b: KeyholeIcon,
  github: GithubLogoIcon,
  google: GoogleLogoIcon,
  groq: KeyholeIcon,
  http: KeyholeIcon,
  notion: NotionLogoIcon,
  nvidia: KeyholeIcon,
  discord: DiscordLogoIcon,
  d_id: KeyholeIcon,
  google_maps: GoogleLogoIcon,
  jina: KeyholeIcon,
  ideogram: KeyholeIcon,
  linear: KeyholeIcon,
  medium: MediumLogoIcon,
  mem0: KeyholeIcon,
  ollama: KeyholeIcon,
  openai: KeyholeIcon,
  openweathermap: KeyholeIcon,
  open_router: KeyholeIcon,
  llama_api: KeyholeIcon,
  pinecone: KeyholeIcon,
  enrichlayer: KeyholeIcon,
  slant3d: KeyholeIcon,
  screenshotone: KeyholeIcon,
  smtp: KeyholeIcon,
  replicate: KeyholeIcon,
  reddit: KeyholeIcon,
  fal: KeyholeIcon,
  revid: KeyholeIcon,
  twitter: TwitterLogoIcon,
  unreal_speech: KeyholeIcon,
  exa: KeyholeIcon,
  hubspot: KeyholeIcon,
  smartlead: KeyholeIcon,
  todoist: KeyholeIcon,
  zerobounce: KeyholeIcon,
};

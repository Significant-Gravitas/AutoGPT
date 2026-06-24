import { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { getHostFromUrl } from "@/lib/utils/url";
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
  provider: string,
  schema?: BlockIOCredentialsSubSchema,
  discriminatorValue?: string,
) => {
  const filtered =
    credentials?.filter((credential) => {
      // First filter by provider
      if (provider !== credential.provider) {
        return false;
      }

      // Check if credential type is supported by this block
      if (schema && !schema.credentials_types.includes(credential.type)) {
        return false;
      }

      // Filter OAuth credentials that have sufficient scopes for this block
      if (credential.type === "oauth2" && schema?.credentials_scopes) {
        const credentialScopes = new Set(credential.scopes || []);
        const requiredScopes = new Set(schema.credentials_scopes);
        const hasAllScopes = [...requiredScopes].every((scope) =>
          credentialScopes.has(scope),
        );
        if (!hasAllScopes) {
          return false;
        }
      }

      // Filter host_scoped credentials by host matching
      if (credential.type === "host_scoped") {
        if (!discriminatorValue) {
          return false;
        }
        const hostFromUrl = getHostFromUrl(discriminatorValue);
        return hostFromUrl === credential.host;
      }

      return true;
    }) ?? [];
  return {
    credentials: filtered,
    exists: filtered.length > 0,
  };
};

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

export const getDiscriminatorValue = (
  formData: Record<string, any>,
  schema: BlockIOCredentialsSubSchema,
): string | undefined => {
  const discriminator = schema.discriminator;
  const discriminatorValues = schema.discriminator_values;

  return [
    discriminator ? formData[discriminator] : null,
    ...(discriminatorValues || []),
  ].find(Boolean);
};

export const getCredentialProviderFromSchema = (
  formData: Record<string, any>,
  schema: BlockIOCredentialsSubSchema,
) => {
  const discriminator = schema.discriminator;
  const discriminatorMapping = schema.discriminator_mapping;
  const providers = schema.credentials_provider;

  const discriminatorValue = getDiscriminatorValue(formData, schema);

  const discriminatedProvider = discriminatorMapping
    ? discriminatorMapping[discriminatorValue ?? ""]
    : null;

  if (providers.length > 1) {
    if (!discriminator) {
      throw new Error(
        "Multi-provider credential input requires discriminator!",
      );
    }
    if (!discriminatedProvider) {
      console.warn(
        `Missing discriminator value from '${discriminator}': ` +
          "hiding credentials input until it is set.",
      );
      return null;
    }
    return discriminatedProvider;
  } else {
    return providers[0];
  }
};

import { KeyIcon } from "@phosphor-icons/react";
import { NotionLogoIcon } from "@radix-ui/react-icons";
import {
  FaDiscord,
  FaGithub,
  FaGoogle,
  FaHubspot,
  FaMedium,
  FaTwitter,
} from "react-icons/fa";

export const fallbackIcon = KeyIcon;

export const providerIcons: Partial<
  Record<string, React.FC<{ className?: string }>>
> = {
  aiml_api: fallbackIcon,
  anthropic: fallbackIcon,
  apollo: fallbackIcon,
  e2b: fallbackIcon,
  github: FaGithub,
  google: FaGoogle,
  groq: fallbackIcon,
  http: fallbackIcon,
  notion: NotionLogoIcon,
  nvidia: fallbackIcon,
  discord: FaDiscord,
  d_id: fallbackIcon,
  google_maps: FaGoogle,
  jina: fallbackIcon,
  ideogram: fallbackIcon,
  linear: fallbackIcon,
  medium: FaMedium,
  mem0: fallbackIcon,
  ollama: fallbackIcon,
  openai: fallbackIcon,
  openweathermap: fallbackIcon,
  open_router: fallbackIcon,
  llama_api: fallbackIcon,
  pinecone: fallbackIcon,
  enrichlayer: fallbackIcon,
  slant3d: fallbackIcon,
  screenshotone: fallbackIcon,
  smtp: fallbackIcon,
  replicate: fallbackIcon,
  reddit: fallbackIcon,
  fal: fallbackIcon,
  revid: fallbackIcon,
  twitter: FaTwitter,
  unreal_speech: fallbackIcon,
  exa: fallbackIcon,
  hubspot: FaHubspot,
  smartlead: fallbackIcon,
  todoist: fallbackIcon,
  zerobounce: fallbackIcon,
};

export type OAuthPopupResultMessage = { message_type: "oauth_popup_result" } & (
  | {
      success: true;
      code: string;
      state: string;
    }
  | {
      success: false;
      message: string;
    }
);

export function getActionButtonText(
  supportsOAuth2: boolean,
  supportsApiKey: boolean,
  supportsUserPassword: boolean,
  supportsHostScoped: boolean,
  hasExistingCredentials: boolean,
): string {
  if (hasExistingCredentials) {
    if (supportsOAuth2) return "Connect another account";
    if (supportsApiKey) return "Use a new API key";
    if (supportsUserPassword) return "Add a new username and password";
    if (supportsHostScoped) return "Add new headers";
    return "Add new credentials";
  } else {
    if (supportsOAuth2) return "Add account";
    if (supportsApiKey) return "Add API key";
    if (supportsUserPassword) return "Add username and password";
    if (supportsHostScoped) return "Add headers";
    return "Add credentials";
  }
}

export function getCredentialDisplayName(
  credential: { title?: string; username?: string },
  displayName: string,
): string {
  return (
    credential.title || credential.username || `Your ${displayName} account`
  );
}

export const OAUTH_TIMEOUT_MS = 5 * 60 * 1000;
export const MASKED_KEY_LENGTH = 15;

export function isSystemCredential(credential: {
  title?: string | null;
  is_system?: boolean;
}): boolean {
  if (credential.is_system === true) return true;
  if (!credential.title) return false;
  const titleLower = credential.title.toLowerCase();
  return (
    titleLower.includes("system") ||
    titleLower.startsWith("use credits for") ||
    titleLower.includes("use credits")
  );
}

export function filterSystemCredentials<
  T extends { title?: string; is_system?: boolean },
>(credentials: T[]): T[] {
  return credentials.filter((cred) => !isSystemCredential(cred));
}

export function getSystemCredentials<
  T extends { title?: string; is_system?: boolean },
>(credentials: T[]): T[] {
  return credentials.filter((cred) => isSystemCredential(cred));
}

import { GlobeSimple, KeyIcon, Lock, Password } from "@phosphor-icons/react";
import { NotionLogoIcon } from "@radix-ui/react-icons";
import {
  FaDiscord,
  FaGithub,
  FaGoogle,
  FaHubspot,
  FaMedium,
  FaTwitter,
} from "react-icons/fa";
import { CredentialsType } from "@/lib/autogpt-server-api/types";

export const fallbackIcon = KeyIcon;

export const providerIcons: Partial<
  Record<string, React.FC<{ className?: string }>>
> = {
  aiml_api: fallbackIcon,
  anthropic: fallbackIcon,
  apollo: fallbackIcon,
  database: fallbackIcon,
  e2b: fallbackIcon,
  github: FaGithub,
  google: FaGoogle,
  groq: fallbackIcon,
  http: fallbackIcon,
  notion: NotionLogoIcon,
  nvidia: fallbackIcon,
  discord: FaDiscord,
  d_id: fallbackIcon,
  elevenlabs: fallbackIcon,
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

export function countSupportedTypes(
  supportsOAuth2: boolean,
  supportsApiKey: boolean,
  supportsUserPassword: boolean,
  supportsHostScoped: boolean,
): number {
  return [
    supportsOAuth2,
    supportsApiKey,
    supportsUserPassword,
    supportsHostScoped,
  ].filter(Boolean).length;
}

export function getSupportedTypes(
  supportsOAuth2: boolean,
  supportsApiKey: boolean,
  supportsUserPassword: boolean,
  supportsHostScoped: boolean,
): CredentialsType[] {
  const types: CredentialsType[] = [];
  if (supportsOAuth2) types.push("oauth2");
  if (supportsApiKey) types.push("api_key");
  if (supportsUserPassword) types.push("user_password");
  if (supportsHostScoped) types.push("host_scoped");
  return types;
}

const CREDENTIAL_TYPE_LABELS: Record<CredentialsType, string> = {
  oauth2: "OAuth",
  api_key: "API Key",
  user_password: "Password",
  host_scoped: "Headers",
};

export function getCredentialTypeLabel(type: CredentialsType): string {
  return CREDENTIAL_TYPE_LABELS[type] ?? type;
}

type CredentialIcon = React.FC<{ className?: string; size?: string | number }>;

export function getCredentialTypeIcon(
  type: CredentialsType,
  provider?: string,
): CredentialIcon {
  if (type === "oauth2" && provider) {
    const icon = providerIcons[provider];
    if (icon) return icon as CredentialIcon;
    return GlobeSimple as CredentialIcon;
  }
  if (type === "api_key") return KeyIcon as CredentialIcon;
  if (type === "user_password") return Password as CredentialIcon;
  if (type === "host_scoped") return Lock as CredentialIcon;
  return KeyIcon as CredentialIcon;
}

export function getActionButtonText(
  supportsOAuth2: boolean,
  supportsApiKey: boolean,
  supportsUserPassword: boolean,
  supportsHostScoped: boolean,
  hasExistingCredentials: boolean,
): string {
  const multipleTypes =
    countSupportedTypes(
      supportsOAuth2,
      supportsApiKey,
      supportsUserPassword,
      supportsHostScoped,
    ) > 1;

  if (multipleTypes) {
    return hasExistingCredentials ? "Add another credential" : "Add credential";
  }

  if (hasExistingCredentials) {
    if (supportsOAuth2) return "Connect another account";
    if (supportsApiKey) return "Use a new API key";
    if (supportsUserPassword) return "Add a new username and password";
    if (supportsHostScoped) return "Update headers";
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
  T extends { title?: string | null; is_system?: boolean },
>(credentials: T[]): T[] {
  return credentials.filter((cred) => !isSystemCredential(cred));
}

export function getSystemCredentials<
  T extends { title?: string | null; is_system?: boolean },
>(credentials: T[]): T[] {
  return credentials.filter((cred) => isSystemCredential(cred));
}

export type DeleteResult =
  | { deleted: true }
  | { deleted: false; need_confirmation: true; message: string };

export type DeleteState = {
  warningMessage: string | null;
  credentialToDelete: { id: string; title: string } | null;
  shouldUnselectCurrent: boolean;
};

export async function processCredentialDeletion(
  credentialToDelete: { id: string; title: string },
  selectedCredentialId: string | undefined,
  deleteCredentials: (id: string, force: boolean) => Promise<DeleteResult>,
  force: boolean,
): Promise<DeleteState> {
  const result = await deleteCredentials(credentialToDelete.id, force);

  if (result.deleted) {
    return {
      warningMessage: null,
      credentialToDelete: null,
      shouldUnselectCurrent: selectedCredentialId === credentialToDelete.id,
    };
  }

  if ("need_confirmation" in result && result.need_confirmation) {
    return {
      warningMessage:
        result.message || "This credential is in use. Force delete?",
      credentialToDelete,
      shouldUnselectCurrent: false,
    };
  }

  return {
    warningMessage: null,
    credentialToDelete,
    shouldUnselectCurrent: false,
  };
}

export function findExistingHostCredentials<
  T extends { type: string; id: string; host?: string },
>(credentials: T[], host: string): T[] {
  return credentials.filter(
    (c) => c.type === "host_scoped" && "host" in c && c.host === host,
  );
}

export function hasExistingHostCredential<
  T extends { type: string; host?: string },
>(credentials: T[], host: string): boolean {
  return credentials.some(
    (c) => c.type === "host_scoped" && "host" in c && c.host === host,
  );
}

export type ActionTarget =
  | "type_selector"
  | "oauth"
  | "api_key"
  | "user_password"
  | "host_scoped"
  | null;

export function resolveActionTarget(
  hasMultipleCredentialTypes: boolean,
  supportsOAuth2: boolean,
  supportsApiKey: boolean,
  supportsUserPassword: boolean,
  supportsHostScoped: boolean,
): ActionTarget {
  if (hasMultipleCredentialTypes) return "type_selector";
  if (supportsOAuth2) return "oauth";
  if (supportsApiKey) return "api_key";
  if (supportsUserPassword) return "user_password";
  if (supportsHostScoped) return "host_scoped";
  return null;
}

export type HeaderPair = { key: string; value: string };

export function headerPairsToRecord(
  pairs: HeaderPair[],
): Record<string, string> {
  return pairs.reduce(
    (acc, pair) => {
      if (pair.key.trim() && pair.value.trim()) {
        acc[pair.key.trim()] = pair.value.trim();
      }
      return acc;
    },
    {} as Record<string, string>,
  );
}

export function addHeaderPairToList(pairs: HeaderPair[]): HeaderPair[] {
  return [...pairs, { key: "", value: "" }];
}

export function removeHeaderPairFromList(
  pairs: HeaderPair[],
  index: number,
): HeaderPair[] {
  if (pairs.length <= 1) return pairs;
  return pairs.filter((_, i) => i !== index);
}

export function updateHeaderPairInList(
  pairs: HeaderPair[],
  index: number,
  field: "key" | "value",
  value: string,
): HeaderPair[] {
  const newPairs = [...pairs];
  newPairs[index] = { ...newPairs[index], [field]: value };
  return newPairs;
}

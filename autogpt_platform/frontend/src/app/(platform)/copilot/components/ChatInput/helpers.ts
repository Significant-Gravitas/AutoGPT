import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { filterSystemCredentials } from "@/components/contextual/CredentialsInput/helpers";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";

// Browsers name clipboard screenshots "image.png"; rename so multiple
// pasted images stay distinguishable in the composer and workspace.
const GENERIC_CLIPBOARD_IMAGE_NAME = /^image\.\w+$/i;

export function getFilesFromClipboard(
  clipboardData: DataTransfer | null,
): File[] {
  if (!clipboardData) return [];
  return Array.from(clipboardData.files).map(renameGenericImage);
}

function renameGenericImage(file: File, index: number): File {
  if (
    !file.type.startsWith("image/") ||
    !GENERIC_CLIPBOARD_IMAGE_NAME.test(file.name)
  ) {
    return file;
  }
  const extension = file.name.split(".").pop();
  const stamp = new Date().toISOString().slice(0, 23).replace(/[T:.]/g, "-");
  const suffix = index > 0 ? `-${index + 1}` : "";
  return new File([file], `pasted-image-${stamp}${suffix}.${extension}`, {
    type: file.type,
    lastModified: file.lastModified,
  });
}

export function formatElapsedTime(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
}

export const CARD_ICON_BUTTON_CLASS =
  "size-9 rounded-full border-transparent bg-zinc-950/[0.06] p-0 text-zinc-900 shadow-none transition-[background-color,transform] hover:border-transparent hover:bg-zinc-950/10 hover:text-zinc-900 active:scale-[0.98] aria-expanded:bg-zinc-950/[0.12]";

export const COMPACT_ICON_BUTTON_CLASS =
  "size-8 rounded-lg border-transparent bg-transparent p-0 text-zinc-700 shadow-none hover:border-transparent hover:bg-zinc-100 hover:text-zinc-900 aria-expanded:bg-zinc-100";

export const COMPACT_SEND_BUTTON_CLASS =
  "size-8 rounded-lg border-zinc-900 bg-zinc-900 text-white hover:border-zinc-800 hover:bg-zinc-800 disabled:border-zinc-200 disabled:bg-zinc-200 disabled:text-white disabled:opacity-100";

export const CARD_SEND_BUTTON_CLASS =
  "size-9 rounded-full border-transparent bg-zinc-950 text-white transition-[background-color,transform] hover:border-transparent hover:bg-zinc-800 active:scale-[0.98] disabled:border-transparent disabled:bg-zinc-950/[0.06] disabled:text-zinc-400";

/** Shared by every path that sends a message. `recovery` is the lowercase
 *  clause telling the user where their text went, e.g. "your message is
 *  back in the composer". */
export function describeSendFailure(error: unknown, recovery: string) {
  const reason = error instanceof Error ? error.message.trim() : "";
  return reason
    ? `${reason} — ${recovery}.`
    : `${recovery.charAt(0).toUpperCase()}${recovery.slice(1)}. Try again.`;
}

/** A connected integration the composer can @-mention. `token` is the exact
 *  text inserted into the prompt, e.g. `@GoogleMaps` for Google Maps. */
export interface IntegrationMention {
  provider: string;
  name: string;
  token: string;
}

export interface MentionRange {
  start: number;
  end: number;
}

export function integrationMentionToken(name: string): string {
  return `@${name.replace(/\s+/g, "")}`;
}

/** One mention per provider the user has actually connected: platform
 *  "use credits" credentials are not something a prompt can address, and
 *  the codex login is the OpenAI integration under a different slug. */
export function connectedIntegrationsFromCredentials(
  credentials: CredentialsMetaResponse[],
): IntegrationMention[] {
  const byProvider = new Map<string, IntegrationMention>();
  for (const credential of filterSystemCredentials(credentials)) {
    const provider =
      credential.provider === "codex" ? "openai" : credential.provider;
    if (!provider || byProvider.has(provider)) continue;
    const name = formatProviderName(provider);
    byProvider.set(provider, {
      provider,
      name,
      token: integrationMentionToken(name),
    });
  }
  return Array.from(byProvider.values()).sort((a, b) =>
    a.name.localeCompare(b.name),
  );
}

function normalizeMentionText(text: string): string {
  return text.replace(/\s+/g, "").toLowerCase();
}

export function filterIntegrationMentions(
  integrations: IntegrationMention[],
  query: string,
): IntegrationMention[] {
  const q = normalizeMentionText(query);
  if (!q) return integrations;
  return integrations.filter(
    (integration) =>
      normalizeMentionText(integration.name).includes(q) ||
      normalizeMentionText(integration.provider).includes(q),
  );
}

/** Replaces the `@query` under the caret with the integration's token and
 *  makes sure a space follows it, so typing continues after the mention.
 *  Returns the new text and where the caret belongs. */
export function insertIntegrationMention(
  value: string,
  range: MentionRange,
  integration: IntegrationMention,
): { value: string; caret: number } {
  const rest = value.slice(range.end);
  const separator = /^\s/.test(rest) ? "" : " ";
  return {
    value: value.slice(0, range.start) + integration.token + separator + rest,
    caret: range.start + integration.token.length + 1,
  };
}

import { serializeCredentialMention } from "../CredentialMention/helpers";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { filterSystemCredentials } from "@/components/contextual/CredentialsInput/helpers";
import {
  serviceName,
  stripProviderPrefix,
} from "@/components/contextual/IntegrationsPanel/helpers";

// Browsers name clipboard screenshots "image.png"; rename so multiple
// pasted images stay distinguishable in the composer and workspace.
const GENERIC_CLIPBOARD_IMAGE_NAME = /^image\.\w+$/i;

export function getFilesFromClipboard(
  clipboardData: DataTransfer | null,
): File[] {
  if (!clipboardData) return [];
  return Array.from(clipboardData.files).map((file, index) =>
    renameGenericImage(file, index, "pasted-image"),
  );
}

/** Mobile browsers hand every camera capture over as "image.jpg"; rename so
 *  photos taken in a row stay distinguishable in the composer and workspace. */
export function nameCapturedPhotos(files: File[]): File[] {
  return files.map((file, index) => renameGenericImage(file, index, "photo"));
}

function renameGenericImage(file: File, index: number, prefix: string): File {
  if (
    !file.type.startsWith("image/") ||
    !GENERIC_CLIPBOARD_IMAGE_NAME.test(file.name)
  ) {
    return file;
  }
  const extension = file.name.split(".").pop();
  const stamp = new Date().toISOString().slice(0, 23).replace(/[T:.]/g, "-");
  const suffix = index > 0 ? `-${index + 1}` : "";
  return new File([file], `${prefix}-${stamp}${suffix}.${extension}`, {
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
  // TODO(tokens): no fitting role for bg-zinc-950/[0.06] #09090B; #09090B is foreground (a text colour) and card-foreground (a text colour); no fitting role for hover:bg-zinc-950/10 #09090B; #09090B is foreground (a text colour) and card-foreground (a text colour); no fitting role for aria-expanded:bg-zinc-950/[0.12] #09090B; #09090B is foreground (a text colour) and card-foreground (a text colour)
  "size-9 rounded-full border-transparent bg-zinc-950/[0.06] p-0 text-foreground-strong shadow-none transition-[background-color,transform] hover:border-transparent hover:bg-zinc-950/10 hover:text-foreground-strong active:scale-[0.98] aria-expanded:bg-zinc-950/[0.12]";

export const COMPACT_ICON_BUTTON_CLASS =
  "size-8 rounded-lg border-transparent bg-transparent p-0 text-foreground-body shadow-none hover:border-transparent hover:bg-surface-hover hover:text-foreground-strong aria-expanded:bg-surface-hover";

export const COMPACT_SEND_BUTTON_CLASS =
  "size-8 rounded-lg border-action-hover bg-action-hover text-action-foreground hover:border-action hover:bg-action disabled:border-border-base disabled:bg-surface-active disabled:text-action-foreground disabled:opacity-100";

export const CARD_SEND_BUTTON_CLASS =
  // TODO(tokens): no fitting role for bg-zinc-950 #09090B; #09090B is foreground (a text colour) and card-foreground (a text colour); no fitting role for disabled:bg-zinc-950/[0.06] #09090B; #09090B is foreground (a text colour) and card-foreground (a text colour)
  "size-9 rounded-full border-transparent bg-zinc-950 text-action-foreground transition-[background-color,transform] hover:border-transparent hover:bg-action active:scale-[0.98] disabled:border-transparent disabled:bg-zinc-950/[0.06] disabled:text-foreground-subtle";

/** Shared by every path that sends a message. `recovery` is the lowercase
 *  clause telling the user where their text went, e.g. "your message is
 *  back in the composer". */
export function describeSendFailure(error: unknown, recovery: string) {
  const reason = error instanceof Error ? error.message.trim() : "";
  return reason
    ? `${reason} — ${recovery}.`
    : `${recovery.charAt(0).toUpperCase()}${recovery.slice(1)}. Try again.`;
}

export interface IntegrationMention {
  credentialId: string;
  provider: string;
  providerName: string;
  name: string;
  username: string | null;
  token: string;
}

export interface MentionRange {
  start: number;
  end: number;
}

export function connectedIntegrationsFromCredentials<
  T extends Pick<
    CredentialsMetaResponse,
    "id" | "provider" | "title" | "username" | "service" | "service_name"
  >,
>(credentials: T[]): IntegrationMention[] {
  const accounts = filterSystemCredentials(credentials)
    .filter((credential) => credential.provider)
    .map((credential) => {
      const provider = credential.provider;
      const providerName = serviceName(credential);
      const title = stripProviderPrefix(
        credential.title?.trim() ?? "",
        provider,
      ).trim();
      const name = title || credential.username?.trim() || providerName;
      const account = {
        credentialId: credential.id,
        provider,
        providerName,
        name,
        username: credential.username,
      };
      return { ...account, token: serializeCredentialMention(account) };
    });
  return accounts.sort(
    (a, b) =>
      a.name.localeCompare(b.name) ||
      a.credentialId.localeCompare(b.credentialId),
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
      normalizeMentionText(integration.provider).includes(q) ||
      normalizeMentionText(integration.providerName).includes(q) ||
      normalizeMentionText(integration.username ?? "").includes(q),
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

/** Phone-sized send button: the default 32px target is too small for a thumb. */
export const MOBILE_SEND_BUTTON_CLASS = "max-md:size-10";

// Every focus the composer takes by itself goes through here, so none of
// them can close a menu the user has open.
export function focusComposer(inputId: string) {
  const input = document.getElementById(inputId);
  if (isFocusInOtherOverlay(input)) return;
  input?.focus();
}

// One on its way out (a sheet closing after a chat is picked) still holds
// focus through its exit animation, but no longer counts.
const OVERLAY_SELECTOR = ["dialog", "alertdialog", "menu"]
  .map((role) => `[role="${role}"]:not([data-state="closed"])`)
  .join(", ");

// Whether focus sits in a menu or dialog the composer is not part of: one
// the user opened while the page was still loading. Taking focus from it
// would close it (Radix dismisses on focus-out).
function isFocusInOtherOverlay(input: HTMLElement | null) {
  const overlay = document.activeElement?.closest(OVERLAY_SELECTOR);
  return Boolean(overlay && input && !overlay.contains(input));
}

export type NoticeAction =
  | { kind: "link"; label: string; href: string }
  | { kind: "resume"; label: string }
  | { kind: "reconnect"; label: string; provider: string }
  | { kind: "run_less_often"; label: string };

/** The note a skipped or failing scheduled chat leaves in its chat. */
export interface ChatNotice {
  reason: string;
  headline: string;
  body: string;
  /** What Resume takes: the routine's id, or the follow-up's schedule id. */
  scheduleKey: string | null;
  actions: NoticeAction[];
}

const NOTICE_METADATA_KIND = "scheduled_chat_notice";
const BILLING_PATH = "/settings/billing";

function asString(value: unknown) {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function toAction(value: unknown): NoticeAction | null {
  if (!value || typeof value !== "object") return null;
  const action = value as Record<string, unknown>;
  const label = asString(action.label);
  if (!label) return null;
  if (action.kind === "resume") return { kind: "resume", label };
  if (action.kind === "run_less_often") return { kind: "run_less_often", label };
  if (action.kind === "reconnect") {
    const provider = asString(action.provider);
    return provider ? { kind: "reconnect", label, provider } : null;
  }
  const href = asString(action.href);
  // In-app paths only; a backslash reads as a slash, so /\host leaves too.
  if (action.kind !== "link" || !href) return null;
  if (!href.startsWith("/") || href.startsWith("//") || href.includes("\\")) return null;
  return { kind: "link", label, href };
}

/** A stored row's `metadata` as a notice, or null for any other row. */
export function parseNotice(metadata: unknown): ChatNotice | null {
  if (!metadata || typeof metadata !== "object") return null;
  const meta = metadata as Record<string, unknown>;
  if (meta.kind !== NOTICE_METADATA_KIND) return null;
  const headline = asString(meta.headline);
  const body = asString(meta.body);
  if (!headline || !body) return null;

  const scheduleKey = asString(meta.schedule_key);
  const actions = Array.isArray(meta.actions)
    ? meta.actions
        .map(toAction)
        .filter((action): action is NoticeAction => action !== null)
        // Resume and Run less often need to know which scheduled chat.
        .filter(
          (action) =>
            (action.kind !== "resume" && action.kind !== "run_less_often") ||
            scheduleKey !== null,
        )
    : [];

  return { reason: asString(meta.reason) ?? "usage", headline, body, scheduleKey, actions };
}

/** The notice a converted message carries in its `metadata`, if any. */
export function messageNotice(metadata: unknown): ChatNotice | null {
  if (!metadata || typeof metadata !== "object") return null;
  const notice = (metadata as { notice?: unknown }).notice;
  return notice && typeof notice === "object" ? (notice as ChatNotice) : null;
}

/** Billing has a screen in the app; every other page opens on the web. */
export function noticeLinkTarget(
  href: string,
  frontendURL: string,
): { kind: "plan" } | { kind: "web"; url: string } {
  const path = href.split(/[?#]/)[0].replace(/\/+$/, "");
  if (path === BILLING_PATH) return { kind: "plan" };
  return { kind: "web", url: `${frontendURL}${href}` };
}

/** "Every Monday at 09:00" reads "every Monday at 09:00" mid-sentence. */
export function lowerFirst(text: string) {
  return text.charAt(0).toLowerCase() + text.slice(1);
}

/** Why the cadence route refused, read or change, by status. */
export function cadenceErrorText(status: number | null) {
  if (status === 402) return "Changing this needs an active plan. You can manage your plan on the web.";
  if (status === 400) return "Pick one of the options shown.";
  if (status === 409) return "It's paused. Resume it first.";
  if (status === 404) return "It no longer runs on a schedule this can change.";
  return null;
}

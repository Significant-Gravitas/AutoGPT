export interface NoticeLinkAction {
  kind: "link";
  label: string;
  href: string;
}

export interface NoticeResumeAction {
  kind: "resume";
  label: string;
}

export interface NoticeReconnectAction {
  kind: "reconnect";
  label: string;
  provider: string;
}

export interface NoticeRunLessOftenAction {
  kind: "run_less_often";
  label: string;
}

export type NoticeAction =
  | NoticeLinkAction
  | NoticeResumeAction
  | NoticeReconnectAction
  | NoticeRunLessOftenAction;

export interface ScheduledChatNotice {
  reason: string;
  headline: string;
  body: string;
  // What Resume takes: the routine's id, or the follow-up's schedule id.
  scheduleKey: string | null;
  actions: NoticeAction[];
}

const NOTICE_METADATA_KIND = "scheduled_chat_notice";

function asString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function toAction(value: unknown): NoticeAction | null {
  if (!value || typeof value !== "object") return null;
  const action = value as Record<string, unknown>;
  const label = asString(action.label);
  if (!label) return null;
  if (action.kind === "resume") return { kind: "resume", label };
  if (action.kind === "run_less_often")
    return { kind: "run_less_often", label };
  if (action.kind === "reconnect") {
    const provider = asString(action.provider);
    return provider ? { kind: "reconnect", label, provider } : null;
  }
  const href = asString(action.href);
  // In-app paths only: the card never sends anyone off the platform.
  if (action.kind !== "link" || !href) return null;
  // A backslash reads as a slash to the browser, so /\host leaves too.
  if (!href.startsWith("/") || href.startsWith("//")) return null;
  if (href.includes("\\")) return null;
  return { kind: "link", label, href };
}

/**
 * Read the note a skipped or failing scheduled chat leaves in its chat, from
 * the message's ``metadata`` bag. Returns null for any other message.
 */
export function getScheduledChatNotice(
  value: unknown,
): ScheduledChatNotice | null {
  if (!value || typeof value !== "object") return null;
  const meta = value as Record<string, unknown>;
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

  return {
    reason: asString(meta.reason) ?? "usage",
    headline,
    body,
    scheduleKey,
    actions,
  };
}

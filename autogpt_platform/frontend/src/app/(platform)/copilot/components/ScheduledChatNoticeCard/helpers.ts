export interface NoticeLinkAction {
  kind: "link";
  label: string;
  href: string;
}

export interface ScheduledChatNotice {
  reason: string;
  headline: string;
  body: string;
  actions: NoticeLinkAction[];
}

const NOTICE_METADATA_KIND = "scheduled_chat_notice";

function asString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function toLinkAction(value: unknown): NoticeLinkAction | null {
  if (!value || typeof value !== "object") return null;
  const action = value as Record<string, unknown>;
  const label = asString(action.label);
  const href = asString(action.href);
  // In-app paths only: the card never sends anyone off the platform.
  if (action.kind !== "link" || !label || !href) return null;
  // A backslash reads as a slash to the browser, so /\host leaves too.
  if (!href.startsWith("/") || href.startsWith("//")) return null;
  if (href.includes("\\")) return null;
  return { kind: "link", label, href };
}

/**
 * Read the note a skipped scheduled chat leaves in its chat, from the
 * message's ``metadata`` bag. Returns null for any other message.
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

  const actions = Array.isArray(meta.actions)
    ? meta.actions
        .map(toLinkAction)
        .filter((action): action is NoticeLinkAction => action !== null)
    : [];

  return {
    reason: asString(meta.reason) ?? "usage",
    headline,
    body,
    actions,
  };
}

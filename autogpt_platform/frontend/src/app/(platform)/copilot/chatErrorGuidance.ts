import { CopilotAuthError } from "./helpers";
import { ChatUnavailableError } from "./stream/turnRuntime";

export type ChatErrorAction = "sign-in" | "new-chat";

export interface ChatErrorGuidance {
  /** What happened, and what will work next. */
  title: string;
  action: ChatErrorAction | null;
}

/** Guidance for a failed send or stream; it promises a resend only where one can succeed. */
export function describeChatError(error: Error): ChatErrorGuidance {
  const status = statusOf(error);
  if (
    error instanceof CopilotAuthError ||
    status === 401 ||
    /authentication failed|not authenticated/i.test(error.message)
  )
    return {
      title: "You're signed out. Sign in again to keep chatting.",
      action: "sign-in",
    };
  if (error instanceof ChatUnavailableError || status === 404)
    return {
      title: "This chat no longer exists. Start a new chat to keep going.",
      action: "new-chat",
    };
  if (/prompt is too long/i.test(error.message))
    return {
      title:
        "This conversation is too long to continue. Start a new chat to keep going.",
      action: "new-chat",
    };
  if (status === 403)
    return {
      title:
        "You don't have access to this, so sending it again won't help. The reason is below.",
      action: null,
    };
  if (status === 429 || /usage limit/i.test(error.message))
    return {
      title: "You've reached a usage limit, so this message wasn't sent.",
      action: null,
    };
  if (
    status !== null &&
    status >= 400 &&
    status < 500 &&
    !RETRYABLE_4XX.has(status)
  )
    return {
      title:
        "This message wasn't accepted, so sending it again won't help. The reason is below.",
      action: null,
    };
  if (status !== null && status >= 500)
    return {
      title:
        "The server had a problem. Please try sending your message again in a moment.",
      action: null,
    };
  if (error.name === "TypeError")
    return {
      title:
        "The connection dropped before your message was sent. Check your connection and send it again.",
      action: null,
    };
  return {
    title: "Something went wrong. Please try sending your message again.",
    action: null,
  };
}

// A timeout, or a conflict with a turn still running, can pass on a resend.
const RETRYABLE_4XX = new Set([408, 409, 425]);

function statusOf(error: Error) {
  const status = (error as { status?: unknown }).status;
  return typeof status === "number" ? status : null;
}

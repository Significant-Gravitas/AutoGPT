import { CopilotAuthError } from "./helpers";
import {
  ChatUnavailableError,
  isUnexpectedRefusal,
  MessageTooLongError,
} from "./stream/turnRuntime";

export type ChatErrorAction = "sign-in" | "new-chat" | "plans";

export interface ChatErrorGuidance {
  /** What happened, and what will work next. */
  title: string;
  action: ChatErrorAction | null;
  /** A small muted line under the guidance. */
  note?: string;
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
  if (error instanceof MessageTooLongError)
    return {
      title: "Your message is too long. Shorten it and send again.",
      action: null,
    };
  if (/prompt is too long/i.test(error.message))
    return {
      title:
        "This conversation is too long to continue. Start a new chat to keep going.",
      action: "new-chat",
    };
  if (status === 402)
    return {
      title: "Sending needs an active plan, so this message wasn't sent.",
      action: "plans",
    };
  if (status === 429 || /usage limit/i.test(error.message))
    return {
      title: "You've reached a usage limit, so this message wasn't sent.",
      action: null,
    };
  // The app only sends what the server accepts, so this is our bug.
  if (status !== null && isUnexpectedRefusal(status))
    return {
      title:
        "The request was refused. Reload the page and try again; if it keeps happening, contact support.",
      action: null,
      note: "The system maintainer has been notified of the issue",
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

function statusOf(error: Error) {
  const status = (error as { status?: unknown }).status;
  return typeof status === "number" ? status : null;
}

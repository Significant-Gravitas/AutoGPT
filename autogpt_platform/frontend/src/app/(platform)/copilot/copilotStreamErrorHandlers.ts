import { toast } from "@/components/molecules/Toast/use-toast";

import {
  describeProviderFailure,
  parseProviderFailure,
  type ProviderFailure,
} from "./providerFailure";

/**
 * Parses a backend-encoded error code from an `errorText` payload.
 *
 * The AI-SDK SSE protocol enforces `z.strictObject({type, errorText})`
 * on StreamError frames, so the backend cannot attach a top-level `code`
 * field. Instead it prefixes the message with `[code:<id>] <msg>` and
 * this helper extracts it client-side.
 */
export function parseBackendErrorCode(raw: string): {
  code: string | null;
  message: string;
} {
  const match = raw.match(/^\s*\[code:([a-z0-9_]+)\]\s*(.*)$/is);
  if (!match) return { code: null, message: raw };
  return { code: match[1], message: match[2].trim() };
}

/**
 * User-facing toast copy for each backend error code we surface.
 * `description` defaults to the backend's error message when provided;
 * `fallbackDescription` is used when the backend sends only the code.
 */
const TOAST_BY_BACKEND_CODE: Record<
  string,
  { title: string; fallbackDescription: string }
> = {
  idle_timeout: {
    title: "Your expert stopped responding",
    fallbackDescription:
      "A tool call got stuck and the session timed out. Press Try Again to resume.",
  },
  tool_stalled: {
    title: "A tool call is taking too long",
    fallbackDescription:
      "The assistant is waiting on a tool that hasn't responded. Press Try Again to restart.",
  },
  transient_api_error: {
    title: "Connection hiccup",
    fallbackDescription:
      "We hit a temporary error talking to the model. Press Try Again to continue.",
  },
  circuit_breaker_empty_tool_calls: {
    title: "Your expert paused",
    fallbackDescription:
      "The assistant made too many empty tool calls in a row and was paused. Press Try Again to continue.",
  },
  all_attempts_exhausted: {
    title: "Conversation too long",
    fallbackDescription:
      "We couldn't fit this chat's history into the model after several attempts. Start a new chat or clear some history.",
  },
  sdk_stream_error: {
    title: "Your expert ran into an error",
    fallbackDescription:
      "Something went wrong while the assistant was responding. Press Try Again to retry.",
  },
  sdk_error: {
    title: "Your expert ran into an error",
    fallbackDescription:
      "The assistant couldn't complete this turn. Press Try Again to retry.",
  },
  max_budget_exhausted: {
    title: "Turn budget reached",
    fallbackDescription:
      "This turn reached its spending limit. Send a follow-up to continue with a smaller scope. If your account usage limit is also reached, wait for it to reset.",
  },
};

/** Fallback toast shown for any `[code:X]` we don't have specific copy for. */
const GENERIC_BACKEND_TOAST = {
  title: "Your expert ran into a problem",
  fallbackDescription:
    "The assistant stopped unexpectedly. Press Try Again to retry.",
};

/**
 * Extract the human-readable error detail. FastAPI typically wraps 4xx
 * responses in `{"detail": "..."}` — if the SDK surfaced that as JSON in
 * `error.message`, unwrap to the nested string; otherwise use the raw message.
 */
function extractErrorDetail(error: Error): string {
  try {
    const parsed = JSON.parse(error.message) as unknown;
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      "detail" in parsed &&
      typeof (parsed as { detail: unknown }).detail === "string"
    ) {
      return (parsed as { detail: string }).detail;
    }
  } catch {
    // Not JSON — use message as-is
  }
  return error.message;
}

/**
 * A typed provider-failure envelope, when FastAPI's `{"detail": ...}` wraps
 * an object instead of a string (e.g. the 429 raised for the platform usage
 * cap). `extractErrorDetail` above only unwraps string details, so a
 * structured refusal used to fall through to substring guessing and never
 * reached the "switch connection" UI. Checked separately so the string path
 * stays untouched for every other error shape.
 */
function extractProviderFailureDetail(error: Error): ProviderFailure | null {
  try {
    const parsed = JSON.parse(error.message) as unknown;
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      "detail" in parsed &&
      typeof (parsed as { detail: unknown }).detail === "object" &&
      (parsed as { detail: unknown }).detail !== null
    ) {
      return parseProviderFailure((parsed as { detail: unknown }).detail);
    }
  } catch {
    // Not JSON
  }
  return null;
}

/**
 * Where a usage limit was refused. The two limits do not share an answer:
 *
 * - `"admission"`: the backend refused the turn before the stream opened,
 *   which only our own budget does. The envelope came in the HTTP error body.
 * - `"provider"`: the connection the turn ran on refused it mid-turn. The
 *   envelope rode the live stream.
 *
 * Told apart by origin rather than by `authProvider` because a self-host
 * runs its own OpenRouter or local gateway on the "platform" route, so its
 * upstream 429 carries the same `authProvider` as our admission cap.
 */
export type UsageLimitOrigin = "admission" | "provider";

interface HandleStreamErrorArgs {
  error: Error;
  onRateLimit: (
    message: string,
    providerFailure?: ProviderFailure,
    origin?: UsageLimitOrigin,
  ) => void;
  onReconnect: () => void;
  isUserStoppingRef: React.MutableRefObject<boolean>;
  /**
   * The typed envelope, when this turn sent one. It is the only source here
   * that actually knows what happened; every branch below it is inference
   * from error text.
   */
  providerFailure?: ProviderFailure | null;
}

/**
 * Process a stream error from `useChat.onError`. Surfaces the right toast
 * (or rate-limit UI), and decides whether to retry via reconnect.
 *
 * Dispatch order (exclusive branches):
 *  0. A typed provider-failure envelope → its own copy.
 *  1. `[code:<id>]` backend prefix → curated or generic backend toast.
 *  2. Legacy `usage limit` substring → rate-limit UI via `onRateLimit`.
 *  3. Legacy 401 / auth failure → auth-error toast.
 *  4. TypeError / AbortError / "connection interrupted" → reconnect.
 *  5. Anything else silently falls through (the AI-SDK also surfaces
 *     `error` into the hook's `error` return, which drives the inline
 *     error banner).
 */
export function handleStreamError({
  error,
  onRateLimit,
  onReconnect,
  isUserStoppingRef,
  providerFailure: streamedProviderFailure,
}: HandleStreamErrorArgs): void {
  const errorDetail = extractErrorDetail(error);
  // The live stream's typed envelope wins when present; otherwise recover
  // one from a structured 429/etc. body raised before streaming started.
  const providerFailure =
    streamedProviderFailure ?? extractProviderFailureDetail(error);

  // 0. The server said what went wrong, so stop guessing.
  if (providerFailure) {
    const copy = describeProviderFailure(providerFailure);
    if (providerFailure.kind === "usage_limit") {
      // Still routed through the rate-limit path: it restores the composer
      // text for a message the backend refused before persisting, which a
      // toast alone would lose.
      //
      // The failure travels with it, and so does where it was refused, so
      // the caller can tell the two limits apart. They are not the same
      // event: our own credits running out is answered by upgrading a plan
      // with us, and a linked subscription running out is answered by
      // continuing on a different connection. Offering the first for the
      // second asks someone to pay us because OpenAI said no.
      const origin: UsageLimitOrigin = streamedProviderFailure
        ? "provider"
        : "admission";
      onRateLimit(
        `${copy.title}. ${copy.description}`,
        providerFailure,
        origin,
      );
      return;
    }
    toast({
      title: copy.title,
      description: copy.description,
      variant: "destructive",
    });
    return;
  }

  // 1. Coded backend errors take precedence over message-text heuristics.
  const { code: backendCode, message: backendMessage } =
    parseBackendErrorCode(errorDetail);
  if (backendCode) {
    const userToast =
      TOAST_BY_BACKEND_CODE[backendCode] ?? GENERIC_BACKEND_TOAST;
    toast({
      title: userToast.title,
      description: backendMessage || userToast.fallbackDescription,
      variant: "destructive",
    });
    return;
  }

  // 2. Rate limit (FastAPI 429 body contains "usage limit")
  if (errorDetail.toLowerCase().includes("usage limit")) {
    onRateLimit(
      errorDetail || "You've reached your usage limit. Please try again later.",
    );
    return;
  }

  // 3. Authentication failures (from getCopilotAuthHeaders or 401 responses)
  const isAuthError =
    errorDetail.includes("Authentication failed") ||
    errorDetail.includes("Unauthorized") ||
    errorDetail.includes("Not authenticated") ||
    errorDetail.toLowerCase().includes("401");
  if (isAuthError) {
    toast({
      title: "Authentication error",
      description: "Your session may have expired. Please sign in again.",
      variant: "destructive",
    });
    return;
  }

  // 4. Transient network / abort — reconnect so the "Try Again" affordance
  // (persisted retryable-error marker on the session) lights up.
  if (isUserStoppingRef.current) return;
  const isNetworkError =
    error.name === "TypeError" || error.name === "AbortError";
  const isTransientApiError = errorDetail.includes("connection interrupted");
  if (isNetworkError || isTransientApiError) {
    onReconnect();
  }
}

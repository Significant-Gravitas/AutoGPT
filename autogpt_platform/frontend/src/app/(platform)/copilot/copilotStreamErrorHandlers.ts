import { toast } from "@/components/molecules/Toast/use-toast";

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
    title: "AutoPilot stopped responding",
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
    title: "AutoPilot paused",
    fallbackDescription:
      "The assistant made too many empty tool calls in a row and was paused. Press Try Again to continue.",
  },
  all_attempts_exhausted: {
    title: "Conversation too long",
    fallbackDescription:
      "We couldn't fit this chat's history into the model after several attempts. Start a new chat or clear some history.",
  },
  sdk_stream_error: {
    title: "AutoPilot ran into an error",
    fallbackDescription:
      "Something went wrong while the assistant was responding. Press Try Again to retry.",
  },
  sdk_error: {
    title: "AutoPilot ran into an error",
    fallbackDescription:
      "The assistant couldn't complete this turn. Press Try Again to retry.",
  },
};

/** Fallback toast shown for any `[code:X]` we don't have specific copy for. */
const GENERIC_BACKEND_TOAST = {
  title: "AutoPilot ran into a problem",
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

interface HandleStreamErrorArgs {
  error: Error;
  onRateLimit: (message: string) => void;
  onReconnect: () => void;
  isUserStoppingRef: React.MutableRefObject<boolean>;
}

/**
 * Process a stream error from `useChat.onError`. Surfaces the right toast
 * (or rate-limit UI), and decides whether to retry via reconnect.
 *
 * Dispatch order (exclusive branches):
 *  1. `usage limit` substring → rate-limit UI via `onRateLimit`.
 *  2. 401 / auth failure → auth-error toast.
 *  3. `[code:<id>]` backend prefix → curated or generic backend toast.
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
}: HandleStreamErrorArgs): void {
  const errorDetail = extractErrorDetail(error);

  // 1. Rate limit (FastAPI 429 body contains "usage limit")
  if (errorDetail.toLowerCase().includes("usage limit")) {
    onRateLimit(
      errorDetail || "You've reached your usage limit. Please try again later.",
    );
    return;
  }

  // 2. Authentication failures (from getCopilotAuthHeaders or 401 responses)
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

  // 3. Coded backend error — show curated or generic toast so backend hangs
  // are never silent.
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

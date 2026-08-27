type ValidationDetailItem = { msg?: unknown };

function readDetail(value: unknown): string | null {
  if (typeof value !== "object" || value === null) return null;
  const detail = (value as { detail?: unknown }).detail;

  if (typeof detail === "string" && detail.length > 0) return detail;

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item: ValidationDetailItem) =>
        typeof item?.msg === "string" ? item.msg : null,
      )
      .filter((msg): msg is string => msg !== null);
    if (messages.length > 0) return messages.join(", ");
  }

  if (typeof detail === "object" && detail !== null) {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === "string" && message.length > 0) {
      const hint = (detail as { hint?: unknown }).hint;
      return typeof hint === "string" && hint.length > 0
        ? `${message} ${hint}`
        : message;
    }
  }

  return null;
}

// Extracts a human-readable message from an error thrown by the API client.
// The mutator builds `ApiError` via `new Error(detail)`, so when the backend
// returns a non-string `detail` (FastAPI 422 array, or a dict), `error.message`
// is coerced to the useless string "[object Object]". Prefer the structured
// `response.detail` and only fall back to `error.message` when it's usable.
export function getOAuthErrorMessage(error: unknown): string {
  if (typeof error === "object" && error !== null) {
    const fromResponse = readDetail((error as { response?: unknown }).response);
    if (fromResponse) return fromResponse;

    const fromError = readDetail(error);
    if (fromError) return fromError;
  }

  if (
    error instanceof Error &&
    error.message &&
    error.message !== "[object Object]"
  ) {
    return error.message;
  }

  return "Something went wrong. Please try again.";
}

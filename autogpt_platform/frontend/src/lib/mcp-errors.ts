type APIErrorPayload = {
  status?: unknown;
  message?: unknown;
  detail?: unknown;
  response?: unknown;
};

export function getErrorStatus(error: unknown): number | null {
  if (typeof error !== "object" || error === null) return null;
  const status = (error as APIErrorPayload).status;
  return typeof status === "number" ? status : null;
}

export function getErrorMessage(
  error: unknown,
  fallback = "Something went wrong. Please try again.",
): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error !== "object" || error === null) return fallback;
  const { message, detail } = error as APIErrorPayload;
  if (typeof detail === "string" && detail) return detail;
  // Routes that need the client to branch on the failure send
  // `detail: {code, message}`; the prose still belongs on screen, so read it
  // out rather than showing the user a stringified object.
  const structured = getStructuredDetail(detail)?.message;
  if (typeof structured === "string" && structured) return structured;
  if (typeof message === "string" && message) return message;
  const value = detail ?? message;
  if (value === undefined || value === null) return fallback;
  return JSON.stringify(value) || fallback;
}

/** The machine-readable `code` a route attached to its error, when it sent one.
 *
 * A failed call arrives in one of two shapes: the plain object
 * `getAPIResponseError` builds from a non-2xx body, or an `ApiError` whose
 * `response` is that body. Both keep the code in the same place inside it. */
export function getErrorCode(error: unknown): string | null {
  if (typeof error !== "object" || error === null) return null;
  const { detail, response } = error as APIErrorPayload;
  const body =
    typeof response === "object" && response !== null
      ? (response as APIErrorPayload).detail
      : undefined;
  const code =
    getStructuredDetail(detail)?.code ?? getStructuredDetail(body)?.code;
  return typeof code === "string" && code ? code : null;
}

function getStructuredDetail(
  detail: unknown,
): { code?: unknown; message?: unknown } | null {
  if (typeof detail !== "object" || detail === null || Array.isArray(detail))
    return null;
  return detail as { code?: unknown; message?: unknown };
}

export function getAPIResponseError(status: number, data: unknown) {
  if (typeof data !== "object" || data === null) {
    return { status, detail: data };
  }
  const detail = "detail" in data ? data.detail : data;
  const message = "message" in data ? data.message : undefined;
  return { status, detail, message };
}

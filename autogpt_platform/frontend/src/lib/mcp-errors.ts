type APIErrorPayload = {
  status?: unknown;
  message?: unknown;
  detail?: unknown;
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
  if (typeof message === "string" && message) return message;
  const value = detail ?? message;
  if (value === undefined || value === null) return fallback;
  return JSON.stringify(value) || fallback;
}

export function getAPIResponseError(status: number, data: unknown) {
  if (typeof data !== "object" || data === null) {
    return { status, detail: data };
  }
  const detail = "detail" in data ? data.detail : data;
  const message = "message" in data ? data.message : undefined;
  return { status, detail, message };
}

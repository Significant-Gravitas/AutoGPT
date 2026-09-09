import { ApiError } from "@/lib/autogpt-server-api/helpers";

/** A 4xx won't heal on retry — fail fast instead of stalling ~7s in backoff. */
export function retryUnlessClientError(
  failureCount: number,
  error: unknown,
): boolean {
  const status = error instanceof ApiError ? error.status : undefined;
  return (
    failureCount < 3 && !(status !== undefined && status >= 400 && status < 500)
  );
}

export function getGraphLoadErrorToast(error: unknown): {
  title: string;
  description: string;
} {
  const status = error instanceof ApiError ? error.status : undefined;
  const description =
    error instanceof Error && error.message
      ? error.message
      : "An unexpected error occurred.";

  if (status === 404) {
    return { title: "Agent not found", description };
  }
  if (status === 401 || status === 403) {
    return { title: "Not authorized to view this agent", description };
  }
  return { title: "Failed to load agent", description };
}

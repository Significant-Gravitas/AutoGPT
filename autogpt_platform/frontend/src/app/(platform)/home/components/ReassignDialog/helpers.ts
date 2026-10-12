import { ApiError } from "@/lib/autogpt-server-api/helpers";

export const RUNNING_MESSAGE =
  "This task is still running. Wait for it to finish, then reassign.";

function serverDetail(error: ApiError): string | undefined {
  const response: unknown = error.response;
  if (typeof response !== "object" || response === null) return undefined;
  const detail = "detail" in response ? response.detail : undefined;
  return typeof detail === "string" ? detail : undefined;
}

export function reassignErrorToast(error: unknown) {
  if (error instanceof ApiError && error.status === 409) {
    return { title: RUNNING_MESSAGE, variant: "destructive" as const };
  }
  return {
    title: "Couldn't reassign the task",
    description: error instanceof ApiError ? serverDetail(error) : undefined,
    variant: "destructive" as const,
  };
}

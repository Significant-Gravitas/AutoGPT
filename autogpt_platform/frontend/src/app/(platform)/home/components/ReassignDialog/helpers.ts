import { ApiError } from "@/lib/autogpt-server-api/helpers";

export const RUNNING_MESSAGE =
  "This task is still running. Wait for it to finish, then reassign.";

// The server's 409s also cover "already reassigned" and "not started".
const RUNNING_DETAIL = "Reassign the task once it stops working";

function serverDetail(error: ApiError): string | undefined {
  const response: unknown = error.response;
  if (typeof response !== "object" || response === null) return undefined;
  const detail = "detail" in response ? response.detail : undefined;
  return typeof detail === "string" ? detail : undefined;
}

export function reassignErrorToast(error: unknown) {
  const detail = error instanceof ApiError ? serverDetail(error) : undefined;
  if (detail === RUNNING_DETAIL) {
    return { title: RUNNING_MESSAGE, variant: "destructive" as const };
  }
  return {
    title: "Couldn't reassign the task",
    description: detail,
    variant: "destructive" as const,
  };
}

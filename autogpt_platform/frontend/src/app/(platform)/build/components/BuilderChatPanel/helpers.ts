import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

export type DeploymentModelStatus =
  | "checking"
  | "available"
  | "missing"
  | "unknown";

interface TransportsQuery {
  isPending: boolean;
  isError: boolean;
  transports: ChatTransportResponse[] | undefined;
}

// Builder sessions only run on this server's own model (the platform
// transport). A connected ChatGPT plan does not open them up, so it doesn't
// count here. "unknown" (the check itself failed) leaves the answer to the
// session request.
export function getDeploymentModelStatus({
  isPending,
  isError,
  transports,
}: TransportsQuery): DeploymentModelStatus {
  if (transports) {
    return transports.some(
      (transport) =>
        transport.auth_provider === "platform" && transport.available,
    )
      ? "available"
      : "missing";
  }
  if (isError) return "unknown";
  return isPending ? "checking" : "unknown";
}

export function isNoModelError(error: unknown) {
  return (
    error instanceof ApiError &&
    error.status === 503 &&
    error.response?.detail === "chat_transport_not_configured"
  );
}

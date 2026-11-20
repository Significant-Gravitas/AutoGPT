import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

export type DeploymentModelStatus =
  | "checking"
  | "available"
  | "missing"
  | "unknown";

interface TransportsQuery {
  isPending: boolean;
  isFetching: boolean;
  transports: ChatTransportResponse[] | undefined;
}

// Builder sessions only run on this server's own model (the platform
// transport). A connected ChatGPT plan does not open them up, so it doesn't
// count here. "unknown" means the check itself failed.
export function getDeploymentModelStatus({
  isPending,
  isFetching,
  transports,
}: TransportsQuery): DeploymentModelStatus {
  if (transports) {
    const isAvailable = transports.some(
      (transport) =>
        transport.auth_provider === "platform" && transport.available,
    );
    if (!isAvailable) return "missing";
    // A cached "available" can be stale (the model may have gone since the
    // panel last opened), and the agent it lets us create can't be taken
    // back, so wait for the refetch.
    return isFetching ? "checking" : "available";
  }
  return isPending || isFetching ? "checking" : "unknown";
}

export function isNoModelError(error: unknown) {
  return (
    error instanceof ApiError &&
    error.status === 503 &&
    error.response?.detail === "chat_transport_not_configured"
  );
}

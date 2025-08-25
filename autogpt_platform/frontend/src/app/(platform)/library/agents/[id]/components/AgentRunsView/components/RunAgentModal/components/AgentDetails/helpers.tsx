import { LibraryAgentStatus } from "@/app/api/__generated__/models/libraryAgentStatus";

export function formatAgentStatus(status: LibraryAgentStatus) {
  const statusMap: Record<string, string> = {
    COMPLETED: "Ready",
    HEALTHY: "Running",
    WAITING: "Run Queued",
    ERROR: "Failed Run",
  };

  return statusMap[status];
}

export function getStatusColor(status: LibraryAgentStatus): string {
  const colorMap: Record<LibraryAgentStatus, string> = {
    COMPLETED: "bg-blue-300",
    HEALTHY: "bg-green-300",
    WAITING: "bg-amber-300",
    ERROR: "bg-red-300",
  };

  return colorMap[status] || "bg-gray-300";
}

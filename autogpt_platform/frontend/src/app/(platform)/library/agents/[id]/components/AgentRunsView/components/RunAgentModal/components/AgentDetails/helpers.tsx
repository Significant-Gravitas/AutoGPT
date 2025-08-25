import { LibraryAgentStatus } from "@/app/api/__generated__/models/libraryAgentStatus";

export function formatAgentStatus(status: LibraryAgentStatus) {
  const statusMap: Record<string, string> = {
    COMPLETED: "Ready",
    HEALTHY: "Healthy",
    WAITING: "Waiting",
    ERROR: "Failed",
  };

  return statusMap[status];
}

export function getStatusColor(status: LibraryAgentStatus): string {
  const colorMap: Record<LibraryAgentStatus, string> = {
    COMPLETED: "bg-green-300",
    HEALTHY: "bg-blue-300",
    WAITING: "bg-yellow-300",
    ERROR: "bg-red-300",
  };

  return colorMap[status] || "bg-gray-300";
}

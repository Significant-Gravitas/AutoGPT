import { Agent } from "@/lib/marketplace-api";
import AdminMarketplaceCard from "./AdminMarketplaceCard";
import { ClipboardX } from "lucide-react";

export default function AdminMarketplaceAgentList({
  agents,
}: {
  agents: Agent[];
}) {
  if (agents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-500">
        <ClipboardX size={48} />
        <p className="mt-4 text-lg font-semibold">No agents to review</p>
      </div>
    );
  }

  return (
    <div>
      {agents.map((agent) => (
        <AdminMarketplaceCard agent={agent} key={agent.id} />
      ))}
    </div>
  );
}

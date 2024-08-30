import { Agent } from "@/lib/marketplace-api";
import AdminMarketplaceCard from "./AdminMarketplaceCard";

export default function AdminMarketplaceAgentList({ agents }: { agents: Agent[] }) {
  return (
    <div>
      {agents.map((agent) => (
        <AdminMarketplaceCard agent={agent} key={agent.id} />
      ))}
    </div>
  );
}
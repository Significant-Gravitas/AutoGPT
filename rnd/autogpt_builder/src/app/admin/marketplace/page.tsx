import { withRoleAccess } from "@/lib/withRoleAccess";
import MarketplaceAPI from "@/lib/marketplace-api";
import { Agent } from "@/lib/marketplace-api/";
import React from "react";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";

async function AdminMarketplace() {
  const agents = await getReviewableAgents();
  console.log("Agents:", agents);
  return (
    <div>
      {agents.agents.map((agent) => (
        <AdminMarketplaceCard agent={agent} key={agent.id} />
      )
      )}
    </div>
  );
}


async function AdminMarketplaceCard({ agent }: { agent: Agent }) {
  const approveAgentWithId = approveAgent.bind(null, agent.id);
  const rejectAgentWithId = rejectAgent.bind(null, agent.id);
  return (
    <Card key={agent.id} className="m-3">
      <a href={`/marketplace/${agent.id}`}>{agent.name}</a>
      <ScrollArea>{agent.description}</ScrollArea>
      <form action={approveAgentWithId}>
        <Button type="submit">
          Approve
        </Button>
      </form>
      <form action={rejectAgentWithId}>
        <Button>
          Reject
        </Button>
      </form>
    </Card>
  );
}



async function approveAgent(agentId: string) {
  "use server";
  console.log(`Approving agent ${agentId}`);
}

async function rejectAgent(agentId: string) {
  "use server";
  console.log(`Rejecting agent ${agentId}`);
}

async function getReviewableAgents() {
  'use server';
  const api = new MarketplaceAPI();
  return api.getAgentSubmissions();

}


export default async function AdminDashboardPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminMarketplace = await withAdminAccess(AdminMarketplace);
  return <ProtectedAdminMarketplace />;
}

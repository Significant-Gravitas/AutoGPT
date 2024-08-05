import { Suspense } from "react";
import { notFound } from "next/navigation";
import MarketplaceAPI from "@/lib/marketplace-api";
import { AgentDetailResponse } from "@/lib/marketplace-api";
import AgentDetailContent from "@/components/AgentDetailContent";

async function getAgentDetails(id: string): Promise<AgentDetailResponse> {
  const apiUrl =
    process.env.NEXT_PUBLIC_AGPT_MARKETPLACE_URL ||
    "http://localhost:8001/api/v1/market";
  const api = new MarketplaceAPI(apiUrl);
  try {
    console.log(`Fetching agent details for id: ${id}`);
    const agent = await api.getAgentDetails(id);
    console.log(`Agent details fetched:`, agent);
    return agent;
  } catch (error) {
    console.error(`Error fetching agent details:`, error);
    throw error;
  }
}

export default async function AgentDetailPage({
  params,
}: {
  params: { id: string };
}) {
  let agent: AgentDetailResponse;

  try {
    agent = await getAgentDetails(params.id);
  } catch (error) {
    return notFound();
  }

  return (
    <Suspense fallback={<div>Loading...</div>}>
      <AgentDetailContent agent={agent} />
    </Suspense>
  );
}

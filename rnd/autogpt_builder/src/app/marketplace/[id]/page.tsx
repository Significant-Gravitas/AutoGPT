import { Suspense, useMemo } from 'react';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import MarketplaceAPI from "@/lib/marketplace-api";
import { AgentDetailResponse } from "@/lib/marketplace-api";

async function getAgentDetails(id: string): Promise<AgentDetailResponse> {
  const apiUrl = process.env.AGPT_MARKETPLACE_URL;
  const api = new MarketplaceAPI(apiUrl);
  try {
    console.log(`Fetching agent details for id: ${id}`); // Add logging
    const agent = await api.getAgentDetails(id);
    console.log(`Agent details fetched:`, agent); // Add logging
    return agent;
  } catch (error) {
    console.error(`Error fetching agent details:`, error); // Add error logging
    return notFound();
  }
}

function AgentDetailContent({ agent }: { agent: AgentDetailResponse }) {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">{agent.name}</h1>
        <Link href="/marketplace" className="px-4 py-2 bg-gray-200 rounded-lg">
          Back
        </Link>
      </div>
      <div className="mt-4">
        <p className="text-gray-700">{agent.description}</p>
      </div>
      <div className="mt-4">
        <Link
          href={`/api/marketplace/agent/${agent.id}/download`}
          className="px-4 py-2 bg-gray-200 rounded-lg"
        >
          Download
        </Link>
      </div>
    </div>
  );
}

export default async function AgentDetailPage({ params }: { params: { id: string } }) {
  console.log(`Rendering AgentDetailPage for id: ${params.id}`); // Add logging

  let agent: AgentDetailResponse | null = null;
  let error: Error | null = null;

  try {
    agent = await getAgentDetails(params.id);
  } catch (e) {
    error = e as Error;
    console.error(`Error in AgentDetailPage:`, error);
  }

  if (error) {
    return <div>Error: {error.message}</div>;
  }

  if (!agent) {
    return notFound();
  }

  return (
    <Suspense fallback={<div>Loading...</div>}>
      <AgentDetailContent agent={agent} />
    </Suspense>
  );
}
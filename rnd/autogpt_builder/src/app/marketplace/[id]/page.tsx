import { Suspense } from 'react';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import MarketplaceAPI from "@/lib/marketplace-api";
import { AgentDetailResponse } from "@/lib/marketplace-api";
import { ArrowLeft, Download, Calendar, Tag } from 'lucide-react';
import { Button } from "@/components/ui/button";

async function getAgentDetails(id: string): Promise<AgentDetailResponse> {
  const apiUrl = process.env.AGPT_MARKETPLACE_URL;
  const api = new MarketplaceAPI(apiUrl);
  try {
    console.log(`Fetching agent details for id: ${id}`);
    const agent = await api.getAgentDetails(id);
    console.log(`Agent details fetched:`, agent);
    return agent;
  } catch (error) {
    console.error(`Error fetching agent details:`, error);
    return notFound();
  }
}

function AgentDetailContent({ agent }: { agent: AgentDetailResponse }) {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <Link href="/marketplace" className="inline-flex items-center text-indigo-600 hover:text-indigo-500 mb-8">
        <ArrowLeft className="mr-2" size={20} />
        Back to Marketplace
      </Link>
      <div className="bg-white shadow overflow-hidden sm:rounded-lg">
        <div className="px-4 py-5 sm:px-6">
          <h1 className="text-3xl font-bold text-gray-900">{agent.name}</h1>
          <p className="mt-1 max-w-2xl text-sm text-gray-500">{agent.description}</p>
        </div>
        <div className="border-t border-gray-200 px-4 py-5 sm:p-0">
          <dl className="sm:divide-y sm:divide-gray-200">
            <div className="py-4 sm:py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 flex items-center">
                <Calendar className="mr-2" size={16} />
                Last Updated
              </dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                {new Date(agent.updatedAt).toLocaleDateString()}
              </dd>
            </div>
            <div className="py-4 sm:py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500 flex items-center">
                <Tag className="mr-2" size={16} />
                Categories
              </dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                {agent.categories.join(', ')}
              </dd>
            </div>
          </dl>
        </div>
      </div>
      <div className="mt-8 flex justify-center">
        <Link href={`/api/marketplace/agent/${agent.id}/download`} passHref>
          <Button className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
            <Download className="mr-2" size={20} />
            Download Agent
          </Button>
        </Link>
      </div>
    </div>
  );
}

export default async function AgentDetailPage({ params }: { params: { id: string } }) {
  console.log(`Rendering AgentDetailPage for id: ${params.id}`);

  let agent: AgentDetailResponse | null = null;
  let error: Error | null = null;

  try {
    agent = await getAgentDetails(params.id);
  } catch (e) {
    error = e as Error;
    console.error(`Error in AgentDetailPage:`, error);
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 text-center">
        <h2 className="text-2xl font-semibold text-gray-900">Error</h2>
        <p className="mt-2 text-red-600">{error.message}</p>
        <Link href="/marketplace" className="mt-4 inline-flex items-center text-indigo-600 hover:text-indigo-500">
          <ArrowLeft className="mr-2" size={20} />
          Back to Marketplace
        </Link>
      </div>
    );
  }

  if (!agent) {
    return notFound();
  }

  return (
    <Suspense fallback={
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 text-center">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        <p className="mt-2 text-gray-600">Loading agent details...</p>
      </div>
    }>
      <AgentDetailContent agent={agent} />
    </Suspense>
  );
}
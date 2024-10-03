"use client";
import Link from "next/link";
import { ArrowLeft, Download, Calendar, Tag } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  AgentDetailResponse,
  InstallationLocation,
} from "@/lib/marketplace-api";
import MarketplaceAPI from "@/lib/marketplace-api";
import AutoGPTServerAPI, { GraphCreatable } from "@/lib/autogpt-server-api";
import "@xyflow/react/dist/style.css";
import { makeAnalyticsEvent } from "./actions";

async function downloadAgent(id: string): Promise<void> {
  const api = new MarketplaceAPI();
  try {
    const file = await api.downloadAgentFile(id);
    console.debug(`Agent file downloaded:`, file);

    // Create a Blob from the file content
    const blob = new Blob([file], { type: "application/json" });

    // Create a temporary URL for the Blob
    const url = window.URL.createObjectURL(blob);

    // Create a temporary anchor element
    const a = document.createElement("a");
    a.href = url;
    a.download = `agent_${id}.json`; // Set the filename

    // Append the anchor to the body, click it, and remove it
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    // Revoke the temporary URL
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error(`Error downloading agent:`, error);
    throw error;
  }
}

async function installGraph(id: string): Promise<void> {
  const apiUrl =
    process.env.NEXT_PUBLIC_AGPT_MARKETPLACE_URL ||
    "http://localhost:8015/api/v1/market";
  const api = new MarketplaceAPI(apiUrl);

  const serverAPIUrl = process.env.NEXT_PUBLIC_AGPT_SERVER_API_URL;
  const serverAPI = new AutoGPTServerAPI(serverAPIUrl);
  try {
    console.debug(`Installing agent with id: ${id}`);
    let agent = await api.downloadAgent(id);
    console.debug(`Agent downloaded:`, agent);
    const data: GraphCreatable = {
      id: agent.id,
      version: agent.version,
      is_active: true,
      is_template: false,
      name: agent.name,
      description: agent.description,
      nodes: agent.graph.nodes,
      links: agent.graph.links,
    };
    const result = await serverAPI.createTemplate(data);
    makeAnalyticsEvent({
      event_name: "agent_installed_from_marketplace",
      event_data: {
        marketplace_agent_id: id,
        installed_agent_id: result.id,
        installation_location: InstallationLocation.CLOUD,
      },
    });
    console.debug(`Agent installed successfully`, result);
  } catch (error) {
    console.error(`Error installing agent:`, error);
    throw error;
  }
}

function AgentDetailContent({ agent }: { agent: AgentDetailResponse }) {
  return (
    <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
      <div className="mb-4 flex items-center justify-between">
        <Link
          href="/marketplace"
          className="inline-flex items-center text-indigo-600 hover:text-indigo-500"
        >
          <ArrowLeft className="mr-2" size={20} />
          Back to Marketplace
        </Link>
        <div className="flex space-x-4">
          <Button
            onClick={() => installGraph(agent.id)}
            className="inline-flex items-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            <Download className="mr-2" size={16} />
            Save to Templates
          </Button>
          <Button
            onClick={() => downloadAgent(agent.id)}
            className="inline-flex items-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            <Download className="mr-2" size={16} />
            Download Agent
          </Button>
        </div>
      </div>
      <div className="overflow-hidden bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:px-6">
          <h1 className="text-3xl font-bold text-gray-900">{agent.name}</h1>
          <p className="mt-1 max-w-2xl text-sm text-gray-500">
            {agent.description}
          </p>
        </div>
        <div className="border-t border-gray-200 px-4 py-5 sm:p-0">
          <dl className="sm:divide-y sm:divide-gray-200">
            <div className="py-4 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6 sm:py-5">
              <dt className="flex items-center text-sm font-medium text-gray-500">
                <Calendar className="mr-2" size={16} />
                Last Updated
              </dt>
              <dd className="mt-1 text-sm text-gray-900 sm:col-span-2 sm:mt-0">
                {new Date(agent.updatedAt).toLocaleDateString()}
              </dd>
            </div>
            <div className="py-4 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6 sm:py-5">
              <dt className="flex items-center text-sm font-medium text-gray-500">
                <Tag className="mr-2" size={16} />
                Categories
              </dt>
              <dd className="mt-1 text-sm text-gray-900 sm:col-span-2 sm:mt-0">
                {agent.categories.join(", ")}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  );
}

export default AgentDetailContent;

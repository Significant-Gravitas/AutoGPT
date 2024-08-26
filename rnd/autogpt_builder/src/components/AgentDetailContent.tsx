"use client";

import { useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Download,
  Calendar,
  Tag,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AgentDetailResponse } from "@/lib/marketplace-api";
import dynamic from "next/dynamic";
import { Node, Edge } from "@xyflow/react";
import MarketplaceAPI from "@/lib/marketplace-api";
import AutoGPTServerAPI, { GraphCreatable } from "@/lib/autogpt-server-api";

const ReactFlow = dynamic(
  () => import("@xyflow/react").then((mod) => mod.ReactFlow),
  { ssr: false },
);
const Controls = dynamic(
  () => import("@xyflow/react").then((mod) => mod.Controls),
  { ssr: false },
);
const Background = dynamic(
  () => import("@xyflow/react").then((mod) => mod.Background),
  { ssr: false },
);

import "@xyflow/react/dist/style.css";
import { beautifyString } from "@/lib/utils";

function convertGraphToReactFlow(graph: any): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = graph.nodes.map((node: any) => {
    let label = node.block_id || "Unknown";
    try {
      label = beautifyString(label);
    } catch (error) {
      console.error("Error beautifying node label:", error);
    }

    return {
      id: node.id,
      position: node.metadata.position || { x: 0, y: 0 },
      data: {
        label,
        blockId: node.block_id,
        inputDefault: node.input_default || {},
        ...node, // Include all other node data
      },
      type: "custom",
    };
  });

  const edges: Edge[] = graph.links.map((link: any) => ({
    id: `${link.source_id}-${link.sink_id}`,
    source: link.source_id,
    target: link.sink_id,
    sourceHandle: link.source_name,
    targetHandle: link.sink_name,
    type: "custom",
    data: {
      sourceId: link.source_id,
      targetId: link.sink_id,
      sourceName: link.source_name,
      targetName: link.sink_name,
      isStatic: link.is_static,
    },
  }));

  return { nodes, edges };
}

async function installGraph(id: string): Promise<void> {
  const apiUrl =
    process.env.NEXT_PUBLIC_AGPT_MARKETPLACE_URL ||
    "http://localhost:8001/api/v1/market";
  const api = new MarketplaceAPI(apiUrl);

  const serverAPIUrl = process.env.AGPT_SERVER_API_URL;
  const serverAPI = new AutoGPTServerAPI(serverAPIUrl);
  try {
    console.log(`Installing agent with id: ${id}`);
    let agent = await api.downloadAgent(id);
    console.log(`Agent downloaded:`, agent);
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
    await serverAPI.createTemplate(data);
    console.log(`Agent installed successfully`);
  } catch (error) {
    console.error(`Error installing agent:`, error);
    throw error;
  }
}

function AgentDetailContent({ agent }: { agent: AgentDetailResponse }) {
  const [isGraphExpanded, setIsGraphExpanded] = useState(false);
  const { nodes, edges } = convertGraphToReactFlow(agent.graph);

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
        <Button
          onClick={() => installGraph(agent.id)}
          className="inline-flex items-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
        >
          <Download className="mr-2" size={16} />
          Download Agent
        </Button>
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
        <div className="border-t border-gray-200 px-4 py-5 sm:px-6">
          <button
            className="flex w-full items-center justify-between text-left text-sm font-medium text-indigo-600 hover:text-indigo-500"
            onClick={() => setIsGraphExpanded(!isGraphExpanded)}
          >
            <span>Agent Graph</span>
            {isGraphExpanded ? (
              <ChevronUp size={20} />
            ) : (
              <ChevronDown size={20} />
            )}
          </button>
          {isGraphExpanded && (
            <div className="mt-4" style={{ height: "600px" }}>
              <ReactFlow
                nodes={nodes}
                edges={edges}
                // nodeTypes={nodeTypes}
                // edgeTypes={edgeTypes}
                // connectionLineComponent={ConnectionLine}
                fitView
                attributionPosition="bottom-left"
                nodesConnectable={false}
                nodesDraggable={false}
                zoomOnScroll={false}
                panOnScroll={false}
                elementsSelectable={false}
              >
                <Controls showInteractive={false} />
                <Background />
              </ReactFlow>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AgentDetailContent;

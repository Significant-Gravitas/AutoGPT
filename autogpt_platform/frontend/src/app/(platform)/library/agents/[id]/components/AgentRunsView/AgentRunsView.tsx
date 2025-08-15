"use client";

import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useAgentRunsView } from "./useAgentRunsView";
import { AgentRunsLoading } from "./components/AgentRunsLoading";
import { Button } from "@/components/atoms/Button/Button";
import { Plus } from "@phosphor-icons/react";

export function AgentRunsView() {
  const { response, ready, error, agentId } = useAgentRunsView();

  // Handle loading state
  if (!ready) {
    return <AgentRunsLoading />;
  }

  // Handle errors - check for query error first, then response errors
  if (error || (response && response.status !== 200)) {
    return (
      <ErrorCard
        isSuccess={false}
        responseError={error || undefined}
        httpError={
          response?.status !== 200
            ? {
                status: response?.status,
                statusText: "Request failed",
              }
            : undefined
        }
        context="agent"
        onRetry={() => window.location.reload()}
      />
    );
  }

  // Handle missing data
  if (!response?.data) {
    return (
      <ErrorCard
        isSuccess={false}
        responseError={{ message: "No agent data found" }}
        context="agent"
        onRetry={() => window.location.reload()}
      />
    );
  }

  const agent = response.data;

  return (
    <div className="grid h-screen grid-cols-[25%_85%] gap-4 pt-8">
      {/* Left Sidebar - 30% */}
      <div className="bg-gray-50 p-4">
        <Button variant="primary" size="large" className="w-full">
          <Plus size={20} /> New Run
        </Button>
      </div>

      {/* Main Content - 70% */}
      <div className="p-4">
        <Breadcrumbs
          items={[
            { name: "My Library", link: "/library" },
            { name: agent.name, link: `/library/agents/${agentId}` },
          ]}
        />
        {/* Main content will go here */}
        <div className="mt-4 text-gray-600">Main content area</div>
      </div>
    </div>
  );
}

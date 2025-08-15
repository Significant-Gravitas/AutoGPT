"use client";

import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useAgentRunsView } from "./useAgentRunsView";
import { AgentRunsLoading } from "./components/AgentRunsLoading";

export function AgentRunsView() {
  const { response, ready, error, agentId } = useAgentRunsView();

  // Handle loading state
  if (!ready) {
    return <ErrorCard loadingSlot={<AgentRunsLoading />} />;
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
    <div className="flex flex-col gap-4">
      <Breadcrumbs
        items={[
          { name: "My Library", link: "/library" },
          { name: agent.name, link: `/library/agents/${agentId}` },
        ]}
      />
    </div>
  );
}

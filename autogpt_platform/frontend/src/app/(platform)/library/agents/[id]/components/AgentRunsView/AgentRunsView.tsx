"use client";

import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useAgentRunsView } from "./useAgentRunsView";
import { AgentRunsLoading } from "./components/AgentRunsLoading";
import { Button } from "@/components/atoms/Button/Button";
import { RunAgentModal } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/RunAgentModal/RunAgentModal";
import { PlusIcon } from "@phosphor-icons/react/dist/ssr";

export function AgentRunsView() {
  const { response, ready, error, agentId } = useAgentRunsView();

  if (!ready) {
    return <AgentRunsLoading />;
  }

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
        <RunAgentModal
          triggerSlot={
            <Button variant="primary" size="large" className="w-full">
              <PlusIcon size={20} /> New Run
            </Button>
          }
          agent={agent}
          agentId={agent.id.toString()}
        />
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

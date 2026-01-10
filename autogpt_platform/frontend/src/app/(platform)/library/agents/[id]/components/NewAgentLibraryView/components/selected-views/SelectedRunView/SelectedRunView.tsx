"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import {
  ScrollableTabs,
  ScrollableTabsContent,
  ScrollableTabsList,
  ScrollableTabsTrigger,
} from "@/components/molecules/ScrollableTabs/ScrollableTabs";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";
import { isLargeScreen, useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useEffect } from "react";
import { AgentInputsReadOnly } from "../../modals/AgentInputsReadOnly/AgentInputsReadOnly";
import { LoadingSelectedContent } from "../LoadingSelectedContent";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { RunDetailHeader } from "../RunDetailHeader/RunDetailHeader";
import { SelectedViewLayout } from "../SelectedViewLayout";
import { RunOutputs } from "./components/RunOutputs";
import { RunSummary } from "./components/RunSummary";
import { SelectedRunActions } from "./components/SelectedRunActions/SelectedRunActions";
import { WebhookTriggerSection } from "./components/WebhookTriggerSection";
import { useSelectedRunView } from "./useSelectedRunView";

interface Props {
  agent: LibraryAgent;
  runId: string;
  onSelectRun?: (id: string) => void;
  onClearSelectedRun?: () => void;
  banner?: React.ReactNode;
  onSelectSettings?: () => void;
  selectedSettings?: boolean;
}

export function SelectedRunView({
  agent,
  runId,
  onSelectRun,
  onClearSelectedRun,
  banner,
  onSelectSettings,
  selectedSettings,
}: Props) {
  const { run, preset, isLoading, responseError, httpError } =
    useSelectedRunView(agent.graph_id, runId);

  const breakpoint = useBreakpoint();
  const isLgScreenUp = isLargeScreen(breakpoint);

  const {
    pendingReviews,
    isLoading: reviewsLoading,
    refetch: refetchReviews,
  } = usePendingReviewsForExecution(runId);

  useEffect(() => {
    if (run?.status === AgentExecutionStatus.REVIEW && runId) {
      refetchReviews();
    }
  }, [run?.status, runId, refetchReviews]);

  const withSummary = run?.stats?.activity_status;
  const withReviews = run?.status === AgentExecutionStatus.REVIEW;

  if (responseError || httpError) {
    return (
      <ErrorCard
        responseError={responseError ?? undefined}
        httpError={httpError ?? undefined}
        context="run"
      />
    );
  }

  if (isLoading && !run) {
    return <LoadingSelectedContent agent={agent} />;
  }

  return (
    <div className="flex h-full w-full gap-4">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <SelectedViewLayout
          agent={agent}
          banner={banner}
          onSelectSettings={onSelectSettings}
          selectedSettings={selectedSettings}
        >
          <div className="flex flex-col gap-4">
            <RunDetailHeader agent={agent} run={run} />

            {!isLgScreenUp ? (
              <SelectedRunActions
                agent={agent}
                run={run}
                onSelectRun={onSelectRun}
                onClearSelectedRun={onClearSelectedRun}
              />
            ) : null}

            {preset &&
              agent.trigger_setup_info &&
              preset.webhook_id &&
              preset.webhook && (
                <WebhookTriggerSection
                  preset={preset}
                  triggerSetupInfo={agent.trigger_setup_info}
                />
              )}

            <ScrollableTabs
              defaultValue={withReviews ? "reviews" : "output"}
              className="-mt-2 flex flex-col"
            >
              <ScrollableTabsList className="px-4">
                {withReviews && (
                  <ScrollableTabsTrigger value="reviews">
                    Reviews ({pendingReviews.length})
                  </ScrollableTabsTrigger>
                )}
                {withSummary && (
                  <ScrollableTabsTrigger value="summary">
                    Summary
                  </ScrollableTabsTrigger>
                )}
                <ScrollableTabsTrigger value="output">
                  Output
                </ScrollableTabsTrigger>
                <ScrollableTabsTrigger value="input">
                  Your input
                </ScrollableTabsTrigger>
              </ScrollableTabsList>
              <div className="my-6 flex flex-col gap-6">
                {/* Human-in-the-Loop Reviews Section */}
                {withReviews && (
                  <ScrollableTabsContent value="reviews">
                    <div className="scroll-mt-4">
                      <RunDetailCard>
                        {reviewsLoading ? (
                          <LoadingSpinner size="small" />
                        ) : pendingReviews.length > 0 ? (
                          <PendingReviewsList
                            reviews={pendingReviews}
                            onReviewComplete={refetchReviews}
                            emptyMessage="No pending reviews for this execution"
                          />
                        ) : (
                          <Text variant="body" className="text-zinc-700">
                            No pending reviews for this execution
                          </Text>
                        )}
                      </RunDetailCard>
                    </div>
                  </ScrollableTabsContent>
                )}

                {/* Summary Section */}
                {withSummary && (
                  <ScrollableTabsContent value="summary">
                    <div className="scroll-mt-4">
                      <RunDetailCard
                        title={
                          <div className="flex items-center gap-1">
                            <Text variant="lead-semibold">Summary</Text>
                            <InformationTooltip
                              iconSize={20}
                              description="This AI-generated summary describes how the agent handled your task. It's an experimental feature and may occasionally be inaccurate."
                            />
                          </div>
                        }
                      >
                        <RunSummary run={run} />
                      </RunDetailCard>
                    </div>
                  </ScrollableTabsContent>
                )}

                {/* Output Section */}
                <ScrollableTabsContent value="output">
                  <div className="scroll-mt-4">
                    <RunDetailCard title="Output">
                      {isLoading ? (
                        <div className="text-neutral-500">
                          <LoadingSpinner />
                        </div>
                      ) : run && "outputs" in run ? (
                        <RunOutputs outputs={run.outputs as any} />
                      ) : (
                        <Text variant="body" className="text-neutral-600">
                          No output from this run.
                        </Text>
                      )}
                    </RunDetailCard>
                  </div>
                </ScrollableTabsContent>

                {/* Input Section */}
                <ScrollableTabsContent value="input">
                  <div id="input" className="scroll-mt-4">
                    <RunDetailCard
                      title={
                        <div className="flex items-center gap-1">
                          <Text variant="lead-semibold">Your input</Text>
                          <InformationTooltip
                            iconSize={20}
                            description="This is the input that was provided to the agent for running this task."
                          />
                        </div>
                      }
                    >
                      <AgentInputsReadOnly
                        agent={agent}
                        inputs={run?.inputs}
                        credentialInputs={run?.credential_inputs}
                      />
                    </RunDetailCard>
                  </div>
                </ScrollableTabsContent>
              </div>
            </ScrollableTabs>
          </div>
        </SelectedViewLayout>
      </div>
      {isLgScreenUp ? (
        <div className="max-w-[3.75rem] flex-shrink-0">
          <SelectedRunActions
            agent={agent}
            run={run}
            onSelectRun={onSelectRun}
            onClearSelectedRun={onClearSelectedRun}
          />
        </div>
      ) : null}
    </div>
  );
}

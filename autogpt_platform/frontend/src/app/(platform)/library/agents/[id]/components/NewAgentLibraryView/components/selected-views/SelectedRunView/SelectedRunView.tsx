"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";
import { isLargeScreen, useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { InfoIcon } from "@phosphor-icons/react";
import { useEffect } from "react";
import { AgentInputsReadOnly } from "../../modals/AgentInputsReadOnly/AgentInputsReadOnly";
import { AnchorLinksWrap } from "../AnchorLinksWrap";
import { LoadingSelectedContent } from "../LoadingSelectedContent";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { RunDetailHeader } from "../RunDetailHeader/RunDetailHeader";
import { SelectedViewLayout } from "../SelectedViewLayout";
import { RunOutputs } from "./components/RunOutputs";
import { RunSummary } from "./components/RunSummary";
import { SelectedRunActions } from "./components/SelectedRunActions/SelectedRunActions";
import { WebhookTriggerSection } from "./components/WebhookTriggerSection";
import { useSelectedRunView } from "./useSelectedRunView";

const anchorStyles =
  "border-b-2 border-transparent pb-1 text-sm font-medium text-slate-600 transition-colors hover:text-slate-900 hover:border-slate-900";

interface Props {
  agent: LibraryAgent;
  runId: string;
  onSelectRun?: (id: string) => void;
  onClearSelectedRun?: () => void;
}

export function SelectedRunView({
  agent,
  runId,
  onSelectRun,
  onClearSelectedRun,
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

  function scrollToSection(id: string) {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

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
    return <LoadingSelectedContent agentName={agent.name} agentId={agent.id} />;
  }

  return (
    <div className="flex h-full w-full gap-4">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <SelectedViewLayout agentName={agent.name} agentId={agent.id}>
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

            {/* Navigation Links */}
            <AnchorLinksWrap>
              {withSummary && (
                <button
                  onClick={() => scrollToSection("summary")}
                  className={anchorStyles}
                >
                  Summary
                </button>
              )}
              <button
                onClick={() => scrollToSection("output")}
                className={anchorStyles}
              >
                Output
              </button>
              <button
                onClick={() => scrollToSection("input")}
                className={anchorStyles}
              >
                Your input
              </button>
              {withReviews && (
                <button
                  onClick={() => scrollToSection("reviews")}
                  className={anchorStyles}
                >
                  Reviews ({pendingReviews.length})
                </button>
              )}
            </AnchorLinksWrap>

            {/* Summary Section */}
            {withSummary && (
              <div id="summary" className="scroll-mt-4">
                <RunDetailCard
                  title={
                    <div className="flex items-center gap-2">
                      <Text variant="lead-semibold">Summary</Text>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <InfoIcon
                              size={16}
                              className="cursor-help text-neutral-500 hover:text-neutral-700"
                            />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="max-w-xs">
                              This AI-generated summary describes how the agent
                              handled your task. It&apos;s an experimental
                              feature and may occasionally be inaccurate.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  }
                >
                  <RunSummary run={run} />
                </RunDetailCard>
              </div>
            )}

            {/* Output Section */}
            <div id="output" className="scroll-mt-4">
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

            {/* Input Section */}
            <div id="input" className="scroll-mt-4">
              <RunDetailCard title="Your input">
                <AgentInputsReadOnly
                  agent={agent}
                  inputs={run?.inputs}
                  credentialInputs={run?.credential_inputs}
                />
              </RunDetailCard>
            </div>

            {/* Reviews Section */}
            {withReviews && (
              <div id="reviews" className="scroll-mt-4">
                <RunDetailCard>
                  {reviewsLoading ? (
                    <div className="text-neutral-500">Loading reviews…</div>
                  ) : pendingReviews.length > 0 ? (
                    <PendingReviewsList
                      reviews={pendingReviews}
                      onReviewComplete={refetchReviews}
                      emptyMessage="No pending reviews for this execution"
                    />
                  ) : (
                    <div className="text-neutral-600">
                      No pending reviews for this execution
                    </div>
                  )}
                </RunDetailCard>
              </div>
            )}
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

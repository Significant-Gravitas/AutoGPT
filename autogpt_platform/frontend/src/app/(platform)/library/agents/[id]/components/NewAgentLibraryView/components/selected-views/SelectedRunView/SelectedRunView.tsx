"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";
import { ensureSupabaseClient } from "@/lib/supabase/hooks/helpers";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect } from "react";
import { AgentInputsReadOnly } from "../../modals/AgentInputsReadOnly/AgentInputsReadOnly";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { RunDetailHeader } from "../RunDetailHeader/RunDetailHeader";
import { RunOutputs } from "./components/RunOutputs";
import { useSelectedRunView } from "./useSelectedRunView";

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
  const { run, isLoading, responseError, httpError } = useSelectedRunView(
    agent.graph_id,
    runId,
  );

  const {
    pendingReviews,
    isLoading: reviewsLoading,
    refetch: refetchReviews,
  } = usePendingReviewsForExecution(runId);

  // Tab state management
  const [activeTab, setActiveTab] = useQueryState(
    "tab",
    parseAsString.withDefault("output"),
  );

  // EXPERIMENTAL: Direct API call bypassing proxy
  useEffect(() => {
    if (!runId || !agent.graph_id) return;

    async function makeDirectCall() {
      const supabaseClient = ensureSupabaseClient();
      if (!supabaseClient) {
        console.error("[EXPERIMENTAL] No Supabase client available");
        return;
      }

      const {
        data: { session },
      } = await supabaseClient.auth.getSession();

      if (!session?.access_token) {
        console.error("[EXPERIMENTAL] No session token available");
        return;
      }

      const baseUrl =
        process.env.NEXT_PUBLIC_AGPT_SERVER_URL || "http://localhost:8006/api";
      const url = `${baseUrl}/graphs/${agent.graph_id}/executions/${runId}`;

      console.log("[EXPERIMENTAL] Making direct API call to:", url);
      const startTime = performance.now();

      try {
        const response = await fetch(url, {
          method: "GET",
          credentials: "include",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${session.access_token}`,
          },
        });

        const endTime = performance.now();
        const duration = endTime - startTime;
        console.log(
          `[EXPERIMENTAL] Direct API call completed in ${duration.toFixed(2)}ms`,
          {
            status: response.status,
            statusText: response.statusText,
            duration: `${duration.toFixed(2)}ms`,
            graphId: agent.graph_id,
            runId: runId,
          },
        );

        if (response.ok) {
          const data = await response.json();
          console.log("[EXPERIMENTAL] Direct API response:", data);
        } else {
          const errorText = await response.text();
          console.error("[EXPERIMENTAL] Direct API error:", errorText);
        }
      } catch (error) {
        const endTime = performance.now();
        const duration = endTime - startTime;
        console.error(
          `[EXPERIMENTAL] Direct API call failed after ${duration.toFixed(2)}ms:`,
          error,
        );
      }
    }

    void makeDirectCall();
  }, [runId, agent.graph_id]);

  useEffect(() => {
    if (run?.status === AgentExecutionStatus.REVIEW && runId) {
      refetchReviews();
    }
  }, [run?.status, runId, refetchReviews]);

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
    return (
      <div className="flex-1 space-y-4">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <RunDetailHeader
        agent={agent}
        run={run}
        onSelectRun={onSelectRun}
        onClearSelectedRun={onClearSelectedRun}
      />

      {/* Content */}
      <TabsLine value={activeTab} onValueChange={setActiveTab}>
        <TabsLineList>
          <TabsLineTrigger value="output">Output</TabsLineTrigger>
          <TabsLineTrigger value="input">Your input</TabsLineTrigger>
          {run?.status === AgentExecutionStatus.REVIEW && (
            <TabsLineTrigger value="reviews">
              Reviews ({pendingReviews.length})
            </TabsLineTrigger>
          )}
        </TabsLineList>

        <TabsLineContent value="output">
          <RunDetailCard>
            {isLoading ? (
              <div className="text-neutral-500">Loading…</div>
            ) : run && "outputs" in run ? (
              <RunOutputs outputs={run.outputs as any} />
            ) : (
              <div className="text-neutral-600">No output from this run.</div>
            )}
          </RunDetailCard>
        </TabsLineContent>

        <TabsLineContent value="input">
          <RunDetailCard>
            <AgentInputsReadOnly
              agent={agent}
              inputs={(run as any)?.inputs}
              credentialInputs={(run as any)?.credential_inputs}
            />
          </RunDetailCard>
        </TabsLineContent>

        {run?.status === AgentExecutionStatus.REVIEW && (
          <TabsLineContent value="reviews">
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
          </TabsLineContent>
        )}
      </TabsLine>
    </div>
  );
}

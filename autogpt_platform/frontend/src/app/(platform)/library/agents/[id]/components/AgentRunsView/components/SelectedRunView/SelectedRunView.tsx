"use client";

import React from "react";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { useSelectedRunView } from "./useSelectedRunView";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { RunDetailHeader } from "../RunDetailHeader/RunDetailHeader";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Skeleton } from "@/components/ui/skeleton";
import { AgentInputsReadOnly } from "../AgentInputsReadOnly/AgentInputsReadOnly";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { RunOutputs } from "./components/RunOutputs";

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
      <TabsLine defaultValue="output">
        <TabsLineList>
          <TabsLineTrigger value="output">Output</TabsLineTrigger>
          <TabsLineTrigger value="input">Your input</TabsLineTrigger>
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
      </TabsLine>
    </div>
  );
}

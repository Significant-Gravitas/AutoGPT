"use client";

import React from "react";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { useRunDetails } from "./useRunDetails";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { RunDetailHeader } from "../RunDetailHeader/RunDetailHeader";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Skeleton } from "@/components/ui/skeleton";
import { AgentInputsReadOnly } from "../AgentInputsReadOnly/AgentInputsReadOnly";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";

interface RunDetailsProps {
  agent: LibraryAgent;
  runId: string;
}

export function RunDetails({ agent, runId }: RunDetailsProps) {
  const { run, isLoading, error } = useRunDetails(agent.graph_id, runId);

  if (error) {
    return (
      <ErrorCard
        responseError={
          error
            ? {
                message: String(
                  (error as unknown as { message?: string })?.message ||
                    "Failed to load run",
                ),
              }
            : undefined
        }
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
      <RunDetailHeader agent={agent} run={run} />

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
            ) : !run ||
              !("outputs" in run) ||
              Object.keys(run.outputs || {}).length === 0 ? (
              <div className="text-neutral-600">No output from this run.</div>
            ) : (
              <div className="flex flex-col gap-4">
                {Object.entries(run.outputs).map(([key, values]) => (
                  <div key={key} className="flex flex-col gap-1.5">
                    <label className="text-sm font-medium">{key}</label>
                    {values.map((value, i) => (
                      <p
                        key={i}
                        className="whitespace-pre-wrap break-words text-sm text-neutral-700"
                      >
                        {typeof value === "object"
                          ? JSON.stringify(value, undefined, 2)
                          : String(value)}
                      </p>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </RunDetailCard>
        </TabsLineContent>

        <TabsLineContent value="input">
          <RunDetailCard>
            <AgentInputsReadOnly agent={agent} inputs={(run as any)?.inputs} />
          </RunDetailCard>
        </TabsLineContent>
      </TabsLine>
    </div>
  );
}

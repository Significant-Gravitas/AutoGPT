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
import moment from "moment";
import { Button } from "@/components/atoms/Button/Button";
import { PencilSimpleIcon, TrashIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { RunStatusBadge } from "./components/RunStatusBadge";
import { ScheduleDetails } from "../ScheduleDetails/ScheduleDetails";

interface RunDetailsProps {
  agent: LibraryAgent;
  runId: string;
}

export function RunDetails({ agent, runId }: RunDetailsProps) {
  const isSchedule = runId.startsWith("schedule:");
  const scheduleId = isSchedule ? runId.replace("schedule:", "") : undefined;
  const { run, isLoading, error } = useRunDetails(
    agent.graph_id,
    isSchedule ? "" : runId,
  );

  if (isSchedule && scheduleId) {
    return <ScheduleDetails agent={agent} scheduleId={scheduleId} />;
  }

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
      <div className="flex flex-col gap-6">
        <div className="h-8 w-60 animate-pulse rounded bg-zinc-200" />
        <div className="h-6 w-40 animate-pulse rounded bg-zinc-200" />
        <div className="h-64 w-full animate-pulse rounded bg-zinc-100" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex w-full items-center justify-between">
        <div className="flex w-full flex-col gap-0">
          <div className="flex w-full items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <RunStatusBadge status={run?.status ?? "FAILED"} />
              <Text variant="h3" className="!font-normal">
                {agent.name}
              </Text>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="small"
                as="NextLink"
                href={`/build?flowID=${agent.graph_id}&flowVersion=${agent.graph_version}`}
                target="_blank"
              >
                <PencilSimpleIcon size={16} /> Edit agent
              </Button>
              <Button variant="secondary" size="small">
                <TrashIcon size={16} /> Delete run
              </Button>
            </div>
          </div>
          {run && (
            <Text variant="small" className="mt-1 !text-zinc-600">
              Started {moment(run.started_at).fromNow()}{" "}
              <span className="mx-1 inline-block">|</span> version{" "}
              {run.graph_version}
            </Text>
          )}
        </div>
      </div>

      {/* Content */}
      <TabsLine defaultValue="output">
        <TabsLineList>
          <TabsLineTrigger value="output">Output</TabsLineTrigger>
          <TabsLineTrigger value="input">Your input</TabsLineTrigger>
        </TabsLineList>

        <TabsLineContent value="output">
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
        </TabsLineContent>

        <TabsLineContent value="input">
          <div className="text-neutral-600">Coming soon</div>
        </TabsLineContent>
      </TabsLine>
    </div>
  );
}

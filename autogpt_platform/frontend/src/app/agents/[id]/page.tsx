"use client";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Plus } from "lucide-react";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  GraphExecution,
  Schedule,
  GraphMeta,
  BlockIOSubType,
} from "@/lib/autogpt-server-api";

import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { Button } from "@/components/agptui/Button";
import { AgentRunStatus } from "@/components/agptui/AgentRunStatusChip";
import AgentRunSummaryCard from "@/components/agptui/AgentRunSummaryCard";
import moment from "moment";

const agentRunStatusMap: Record<GraphExecution["status"], AgentRunStatus> = {
  COMPLETED: "success",
  FAILED: "failed",
  QUEUED: "queued",
  RUNNING: "running",
  // TODO: implement "draft"
  // TODO: implement "stopped"
};

export default function AgentRunsPage(): React.ReactElement {
  const { id: agentID }: { id: string } = useParams();
  const router = useRouter();
  const api = useBackendAPI();

  const [agent, setAgent] = useState<GraphMeta | null>(null);
  const [agentRuns, setAgentRuns] = useState<GraphExecution[]>([]);
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [selectedRun, setSelectedRun] = useState<
    GraphExecution | Schedule | null
  >(null);
  const [activeListTab, setActiveListTab] = useState<"runs" | "scheduled">(
    "runs",
  );

  const fetchAgents = useCallback(() => {
    api.getGraph(agentID).then(setAgent);
    api.getGraphExecutions(agentID).then((agentRuns) => {
      setAgentRuns(agentRuns.toSorted((a, b) => b.started_at - a.started_at));

      if (!selectedRun) {
        setSelectedRun(agentRuns[0]);
      }
    });
  }, [api, agentID, selectedRun]);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  const fetchSchedules = useCallback(async () => {
    // TODO: filter in backend
    setSchedules(
      (await api.listSchedules()).filter((s) => s.graph_id == agentID),
    );
  }, [api, agentID]);

  useEffect(() => {
    fetchSchedules();
  }, [fetchSchedules]);

  const removeSchedule = useCallback(
    async (scheduleId: string) => {
      const removedSchedule = await api.deleteSchedule(scheduleId);
      setSchedules(schedules.filter((s) => s.id !== removedSchedule.id));
    },
    [schedules, api],
  );

  /* TODO: use websockets instead of polling */
  useEffect(() => {
    const intervalId = setInterval(() => fetchAgents(), 5000);
    return () => clearInterval(intervalId);
  }, [fetchAgents, agent]);

  const selectedRunStatus: AgentRunStatus = useMemo(
    () =>
      !selectedRun
        ? "draft"
        : "status" in selectedRun
          ? agentRunStatusMap[selectedRun.status]
          : "scheduled",
    [selectedRun],
  );

  const infoStats: { label: string; value: React.ReactNode }[] = useMemo(() => {
    if (!selectedRun) return [];
    return [
      {
        label: "Status",
        value:
          selectedRunStatus.charAt(0).toUpperCase() +
          selectedRunStatus.slice(1),
      },
      ...("started_at" in selectedRun
        ? [
            {
              label: "Started",
              value: `${moment(selectedRun.started_at).fromNow()}, ${moment(selectedRun.started_at).format("HH:mm")}`,
            },
            {
              label: "Duration",
              value: `${moment.duration(selectedRun.duration, "seconds").humanize()}`,
            },
            // { label: "Cost", value: selectedRun.cost },  // TODO: implement cost
          ]
        : [{ label: "Scheduled for", value: selectedRun.next_run_time }]),
    ];
  }, [selectedRun, selectedRunStatus]);

  const agentRunInputs: Record<string, { type: BlockIOSubType; value: any }> =
    useMemo(() => {
      if (!selectedRun) return {};
      // return selectedRun.input; // TODO: implement run input view
      return {
        "Mock Input": { type: "string", value: "Mock Value" },
      };
    }, [selectedRun]);

  const runAgain = useCallback(
    () =>
      api.executeGraph(
        agentID,
        Object.fromEntries(
          Object.entries(agentRunInputs).map(([k, v]) => [k, v.value]),
        ),
      ),
    [api, agentID, agentRunInputs],
  );

  const agentRunOutputs: Record<string, { type: BlockIOSubType; value: any }> =
    useMemo(() => {
      if (
        !selectedRun ||
        !["running", "success", "failed"].includes(selectedRunStatus) ||
        !("output" in selectedRun)
      )
        return {};
      // return selectedRun.output; // TODO: implement run output view
      return {
        "Mock Output": { type: "string", value: "Mock Value" },
      };
    }, [selectedRun, selectedRunStatus]);

  const runActions: { label: string; callback: () => void }[] = useMemo(() => {
    if (!selectedRun) return [];
    return [{ label: "Run again", callback: () => runAgain() }];
  }, [selectedRun, runAgain]);

  const agentActions: { label: string; callback: () => void }[] =
    useMemo(() => {
      if (!agentID) return [];
      return [
        {
          label: "Open in builder",
          callback: () => router.push(`/build?flowID=${agentID}`),
        },
      ];
    }, [agentID, router]);

  if (!agent) {
    /* TODO: implement loading indicators / skeleton page */
    return <span>Loading...</span>;
  }

  return (
    <div className="flex gap-8">
      <aside className="flex w-72 flex-col gap-4">
        <Button className="flex w-full items-center gap-2 py-6">
          <Plus className="h-6 w-6" />
          <span>New run</span>
        </Button>

        <div className="flex gap-2">
          <Badge
            variant={activeListTab === "runs" ? "secondary" : "outline"}
            className="cursor-pointer gap-2"
            onClick={() => setActiveListTab("runs")}
          >
            <span>Runs</span>
            <span className="text-neutral-600">{agentRuns.length}</span>
          </Badge>

          <Badge
            variant={activeListTab === "scheduled" ? "secondary" : "outline"}
            className="cursor-pointer gap-2"
            onClick={() => setActiveListTab("scheduled")}
          >
            <span>Scheduled</span>
            <span className="text-neutral-600">
              {schedules.filter((s) => s.graph_id === agentID).length}
            </span>
          </Badge>
        </div>

        <ScrollArea className="h-[calc(100vh-200px)]">
          <div className="flex flex-col gap-2">
            {activeListTab === "runs"
              ? agentRuns.map((run, i) => (
                  <AgentRunSummaryCard
                    key={i}
                    agentID={run.graph_id}
                    agentRunID={run.execution_id}
                    status={agentRunStatusMap[run.status]}
                    title={agent.name}
                    timestamp={run.started_at}
                    onClick={() => setSelectedRun(run)}
                  />
                ))
              : schedules
                  .filter((schedule) => schedule.graph_id === agentID)
                  .map((schedule, i) => (
                    <AgentRunSummaryCard
                      key={i}
                      agentID={schedule.graph_id}
                      agentRunID={schedule.id}
                      status="scheduled"
                      title={schedule.name}
                      timestamp={schedule.next_run_time} // FIXME
                      onClick={() => setSelectedRun(schedule)}
                    />
                  ))}
          </div>
        </ScrollArea>
      </aside>

      <div className="flex-1">
        <h1 className="mb-8 text-3xl font-medium">
          {agent.name /* TODO: use dynamic/custom run title */}
        </h1>

        <Tabs defaultValue="info">
          <TabsList>
            <TabsTrigger value="info">Info</TabsTrigger>
            <TabsTrigger value="output">Output</TabsTrigger>
            <TabsTrigger value="input">Input</TabsTrigger>
            <TabsTrigger value="rate">Rate</TabsTrigger>
          </TabsList>

          <TabsContent value="info">
            <Card>
              <CardHeader>
                <CardTitle>Info</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex justify-evenly gap-4">
                  {infoStats.map(({ label, value }) => (
                    <div key={label}>
                      <p className="text-sm font-medium text-black">{label}</p>
                      <p className="text-sm text-neutral-600">{value}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="input">
            <Card>
              <CardHeader>
                <CardTitle>Input</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                {Object.entries(agentRunInputs).map(([key, { value }]) => (
                  <div key={key} className="flex flex-col gap-1.5">
                    <label className="text-sm font-medium">{key}</label>
                    <Input
                      defaultValue={value}
                      className="rounded-full"
                      disabled
                    />
                  </div>
                ))}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      <aside className="w-64">
        <div className="flex flex-col gap-8">
          <div className="flex flex-col gap-3">
            <h3 className="text-sm font-medium">Run actions</h3>
            {runActions.map((action, i) => (
              <Button key={i} variant="outline" onClick={action.callback}>
                {action.label}
              </Button>
            ))}
          </div>

          <div className="flex flex-col gap-3">
            <h3 className="text-sm font-medium">Agent actions</h3>
            {agentActions.map((action, i) => (
              <Button key={i} variant="outline" onClick={action.callback}>
                {action.label}
              </Button>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}

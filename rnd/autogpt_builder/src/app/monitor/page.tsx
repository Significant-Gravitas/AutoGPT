"use client";
import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import moment from 'moment';
import {
  ComposedChart,
  DefaultLegendContentProps,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Pencil2Icon } from '@radix-ui/react-icons';
import AutoGPTServerAPI, { Flow, NodeExecutionResult } from '@/lib/autogpt_server_api';
import { cn, hashString } from '@/lib/utils';
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

const Monitor = () => {
  const [flows, setFlows] = useState<Flow[]>([]);
  const [flowRuns, setFlowRuns] = useState<FlowRun[]>([]);
  const [selectedFlow, setSelectedFlow] = useState<Flow | null>(null);
  const [selectedRun, setSelectedRun] = useState<FlowRun | null>(null);

  const api = new AutoGPTServerAPI();

  useEffect(() => fetchFlowsAndRuns(), []);
  useEffect(() => {
    const intervalId = setInterval(() => flows.map(f => refreshFlowRuns(f.id)), 5000);
    return () => clearInterval(intervalId);
  }, []);

  function fetchFlowsAndRuns() {
    // Fetch flow IDs
    api.listFlowIDs()
    .then(flowIDs => {
      Promise.all(flowIDs.map(flowID => {
        refreshFlowRuns(flowID);

        // Fetch flow
        return api.getFlow(flowID);
      }))
      .then(flows => setFlows(flows));
    });
  }

  function refreshFlowRuns(flowID: string) {
    // Fetch flow run IDs
    api.listFlowRunIDs(flowID)
    .then(runIDs => runIDs.map(runID => {
      let run;
      if (
        (run = flowRuns.find(fr => fr.id == runID))
        && !["waiting", "running"].includes(run.status)
      ) {
        return
      }

      // Fetch flow run
      api.getFlowExecutionInfo(flowID, runID)
      .then(execInfo => setFlowRuns(flowRuns => {
        const flowRunIndex = flowRuns.findIndex(fr => fr.id == runID);
        const flowRun = flowRunFromNodeExecutionResults(flowID, runID, execInfo)
        if (flowRunIndex > -1) {
          flowRuns.splice(flowRunIndex, 1, flowRun)
        }
        else {
          flowRuns.push(flowRun)
        }
        return flowRuns
      }));
    }));
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-5 lg:grid-cols-4 xl:grid-cols-10 gap-4">
      <AgentFlowList
        className="md:col-span-2 xl:col-span-3 xxl:col-span-2"
        flows={flows}
        flowRuns={flowRuns}
        selectedFlow={selectedFlow}
        onSelectFlow={f => {
          setSelectedRun(null);
          setSelectedFlow(f.id == selectedFlow?.id ? null : f);
        }}
      />
      <FlowRunsList
        className="md:col-span-3 lg:col-span-2 xl:col-span-3 space-y-4"
        flows={flows}
        runs={
          (
            selectedFlow
              ? flowRuns.filter(v => v.flowID == selectedFlow.id)
              : flowRuns
          )
          .toSorted((a, b) => Number(a.startTime) - Number(b.startTime))
        }
        selectedRun={selectedRun}
        onSelectRun={r => setSelectedRun(r.id == selectedRun?.id ? null : r)}
      />
      <div className="col-span-full xl:col-span-4 xxl:col-span-5">
        {selectedRun && (
          <FlowRunInfo
            flow={selectedFlow || flows.find(f => f.id == selectedRun.flowID)!}
            flowRun={selectedRun}
          />
        ) || selectedFlow && (
          <FlowInfo
            flow={selectedFlow}
            flowRuns={flowRuns.filter(r => r.flowID == selectedFlow.id)}
          />
        ) || (
          <Card className="p-6">
            <FlowRunsStats flows={flows} flowRuns={flowRuns} />
          </Card>
        )}
      </div>
    </div>
  );
};

type FlowRun = {
  id: string
  flowID: string
  status: 'running' | 'waiting' | 'success' | 'failed'
  startTime: number // unix timestamp (ms)
  endTime: number // unix timestamp (ms)
  duration: number // seconds
  totalRunTime: number // seconds

  nodeExecutionResults: NodeExecutionResult[]
};

function flowRunFromNodeExecutionResults(
  flowID: string, runID: string, nodeExecutionResults: NodeExecutionResult[]
): FlowRun {
  // Determine overall status
  let status: 'running' | 'waiting' | 'success' | 'failed' = 'success';
  for (const execution of nodeExecutionResults) {
    if (execution.status === 'FAILED') {
      status = 'failed';
      break;
    } else if (['QUEUED', 'RUNNING'].includes(execution.status)) {
      status = 'running';
      break;
    } else if (execution.status === 'INCOMPLETE') {
      status = 'waiting';
    }
  }

  // Determine aggregate startTime, endTime, and totalRunTime
  const now = Date.now();
  const startTime = Math.min(
    ...nodeExecutionResults.map(ner => ner.add_time.getTime()), now
  );
  const endTime = (
    ['success', 'failed'].includes(status)
      ? Math.max(
        ...nodeExecutionResults.map(ner => ner.end_time?.getTime() || 0), startTime
      )
      : now
  );
  const duration = (endTime - startTime) / 1000;  // Convert to seconds
  const totalRunTime = nodeExecutionResults.reduce((cum, node) => (
    cum + ((node.end_time?.getTime() ?? now) - (node.start_time?.getTime() ?? now))
  ), 0) / 1000;

  return {
    id: runID,
    flowID: flowID,
    status,
    startTime,
    endTime,
    duration,
    totalRunTime,
    nodeExecutionResults: nodeExecutionResults
  };
}

const AgentFlowList = (
  { flows, flowRuns, selectedFlow, onSelectFlow, className }: {
    flows: Flow[],
    flowRuns?: FlowRun[],
    selectedFlow: Flow | null,
    onSelectFlow: (f: Flow) => void,
    className?: string,
  }
) => (
  <Card className={className}>
    <CardHeader>
      <CardTitle>Agents</CardTitle>
    </CardHeader>
    <CardContent>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            {/* <TableHead>Status</TableHead> */}
            {/* <TableHead>Last updated</TableHead> */}
            {flowRuns && <TableHead className="md:hidden lg:table-cell"># of runs</TableHead>}
            {flowRuns && <TableHead>Last run</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {flows
            .map((flow) => {
              let runCount = 0, lastRun: FlowRun | null = null;
              if (flowRuns) {
                const _flowRuns = flowRuns.filter(r => r.flowID == flow.id);
                runCount = _flowRuns.length;
                lastRun = runCount == 0 ? null : _flowRuns.reduce(
                  (a, c) => a.startTime > c.startTime ? a : c
                );
              }
              return { flow, runCount, lastRun };
            })
            .sort((a, b) => {
              if (!a.lastRun && !b.lastRun) return 0;
              if (!a.lastRun) return 1;
              if (!b.lastRun) return -1;
              return b.lastRun.startTime - a.lastRun.startTime;
            })
            .map(({ flow, runCount, lastRun }) => (
              <TableRow
                key={flow.id}
                className="cursor-pointer"
                onClick={() => onSelectFlow(flow)}
                data-state={selectedFlow?.id == flow.id ? "selected" : null}
              >
                <TableCell>{flow.name}</TableCell>
                {/* <TableCell><FlowStatusBadge status={flow.status ?? "active"} /></TableCell> */}
                {/* <TableCell>
                  {flow.updatedAt ?? "???"}
                </TableCell> */}
                {flowRuns && <TableCell className="md:hidden lg:table-cell">{runCount}</TableCell>}
                {flowRuns && (!lastRun ? <TableCell /> :
                <TableCell title={moment(lastRun.startTime).toString()}>
                  {moment(lastRun.startTime).fromNow()}
                </TableCell>)}
              </TableRow>
            ))
          }
        </TableBody>
      </Table>
    </CardContent>
  </Card>
);

const FlowStatusBadge = ({ status }: { status: "active" | "disabled" | "failing" }) => (
  <Badge
    variant="default"
    className={
      status === 'active' ? 'bg-green-500 dark:bg-green-600' :
      status === 'failing' ? 'bg-red-500 dark:bg-red-700' :
      'bg-gray-500 dark:bg-gray-600'
    }
  >
    {status}
  </Badge>
);

const FlowRunsList: React.FC<{
  flows: Flow[];
  runs: FlowRun[];
  className?: string;
  selectedRun?: FlowRun | null;
  onSelectRun: (r: FlowRun) => void;
}> = ({ flows, runs, selectedRun, onSelectRun, className }) => (
  <Card className={className}>
    <CardHeader>
      <CardTitle>Runs</CardTitle>
    </CardHeader>
    <CardContent>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Agent</TableHead>
            <TableHead>Started</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Duration</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <TableRow
              key={run.id}
              className="cursor-pointer"
              onClick={() => onSelectRun(run)}
              data-state={selectedRun?.id == run.id ? "selected" : null}
            >
              <TableCell>{flows.find(f => f.id == run.flowID)!.name}</TableCell>
              <TableCell>{moment(run.startTime).format("HH:mm")}</TableCell>
              <TableCell><FlowRunStatusBadge status={run.status} /></TableCell>
              <TableCell>{formatDuration(run.duration)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </CardContent>
  </Card>
);

const FlowRunStatusBadge: React.FC<{
  status: FlowRun['status'];
  className?: string;
}> = ({ status, className }) => (
  <Badge
    variant="default"
    className={cn(
      status === 'running' ? 'bg-blue-500 dark:bg-blue-700' :
      status === 'waiting' ? 'bg-yellow-500 dark:bg-yellow-600' :
      status === 'success' ? 'bg-green-500 dark:bg-green-600' :
      'bg-red-500 dark:bg-red-700',
      className,
    )}
  >
    {status}
  </Badge>
);

const FlowInfo: React.FC<{
  flow: Flow;
  flowRuns: FlowRun[];
}> = ({ flow, flowRuns }) => {
  return <Card>
    <CardHeader className="flex-row items-center justify-between space-y-0 space-x-3">
      <div>
        <CardTitle>{flow.name}</CardTitle>
        <p className="mt-2">Agent ID: <code>{flow.id}</code></p>
      </div>
      <Link className={buttonVariants({ variant: "outline" })} href={`/build?flowID=${flow.id}`}>
        <Pencil2Icon className="mr-2" /> Edit Agent
      </Link>
    </CardHeader>
    <CardContent>
      <FlowRunsStats
        flows={[flow]}
        flowRuns={flowRuns.filter(r => r.flowID == flow.id)}
      />
    </CardContent>
  </Card>;
};

const FlowRunInfo: React.FC<{
  flow: Flow;
  flowRun: FlowRun;
}> = ({ flow, flowRun }) => {
  if (flowRun.flowID != flow.id) {
    throw new Error(`FlowRunInfo can't be used with non-matching flowRun.flowID and flow.id`)
  }

  return <Card>
    <CardHeader className="flex-row items-center justify-between space-y-0 space-x-3">
      <div>
        <CardTitle>{flow.name}</CardTitle>
        <p className="mt-2">Agent ID: <code>{flow.id}</code></p>
        <p className="mt-1">Run ID: <code>{flowRun.id}</code></p>
      </div>
      <Link className={buttonVariants({ variant: "outline" })} href={`/build?flowID=${flow.id}`}>
        <Pencil2Icon className="mr-2" /> Edit Agent
      </Link>
    </CardHeader>
    <CardContent>
      <p><strong>Status:</strong> <FlowRunStatusBadge status={flowRun.status} /></p>
      <p><strong>Started:</strong> {moment(flowRun.startTime).format('YYYY-MM-DD HH:mm:ss')}</p>
      <p><strong>Finished:</strong> {moment(flowRun.endTime).format('YYYY-MM-DD HH:mm:ss')}</p>
      <p><strong>Duration (run time):</strong> {flowRun.duration} ({flowRun.totalRunTime}) seconds</p>
      {/* <p><strong>Total cost:</strong> €1,23</p> */}
    </CardContent>
  </Card>;
};

const FlowRunsStats: React.FC<{
  flows: Flow[],
  flowRuns: FlowRun[],
  title?: string,
  className?: string,
}> = ({ flows, flowRuns, title, className }) => {
  /* "dateMin": since the first flow in the dataset
   * number > 0: custom date (unix timestamp)
   * number < 0: offset relative to Date.now() (in seconds) */
  const [statsSince, setStatsSince] = useState<number | "dataMin">(-24*3600)
  const statsSinceTimestamp = (  // unix timestamp or null
    typeof(statsSince) == "string"
      ? null
      : statsSince < 0
        ? Date.now() + (statsSince*1000)
        : statsSince
  )
  const filteredFlowRuns = statsSinceTimestamp != null
    ? flowRuns.filter(fr => fr.startTime > statsSinceTimestamp)
    : flowRuns;

  return (
    <div className={className}>
      <div className="flex flex-row items-center justify-between">
        <CardTitle>{ title || "Stats" }</CardTitle>
        <div className="flex space-x-2">
          <Button variant="outline" size="sm" onClick={() => setStatsSince(-2*3600)}>2h</Button>
          <Button variant="outline" size="sm" onClick={() => setStatsSince(-8*3600)}>8h</Button>
          <Button variant="outline" size="sm" onClick={() => setStatsSince(-24*3600)}>24h</Button>
          <Button variant="outline" size="sm" onClick={() => setStatsSince(-7*24*3600)}>7d</Button>
          <Popover>
            <PopoverTrigger asChild>
              <Button variant={"outline"} size="sm">Custom</Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                onSelect={(_, selectedDay) => setStatsSince(selectedDay.getTime())}
                initialFocus
              />
            </PopoverContent>
          </Popover>
          <Button variant="outline" size="sm" onClick={() => setStatsSince("dataMin")}>All</Button>
        </div>
      </div>
      <FlowRunsTimeline flows={flows} flowRuns={flowRuns} dataMin={statsSince} className="mt-3" />
      <hr className="my-4" />
      <div>
        <p><strong>Total runs:</strong> {filteredFlowRuns.length}</p>
        <p>
          <strong>Total run time:</strong> {
            filteredFlowRuns.reduce((total, run) => total + run.totalRunTime, 0)
          } seconds
        </p>
        {/* <p><strong>Total cost:</strong> €1,23</p> */}
      </div>
    </div>
  )
}

const FlowRunsTimeline = (
  { flows, flowRuns, dataMin, className }: {
    flows: Flow[],
    flowRuns: FlowRun[],
    dataMin: "dataMin" | number,
    className?: string,
  }
) => (
  /* TODO: make logarithmic? */
  <ResponsiveContainer width="100%" height={120} className={className}>
    <ComposedChart>
      <XAxis
        dataKey="time"
        type="number"
        domain={[
          typeof(dataMin) == "string"
            ? dataMin
            : dataMin < 0
              ? Date.now() + (dataMin*1000)
              : dataMin,
          Date.now()
        ]}
        allowDataOverflow={true}
        tickFormatter={(unixTime) => {
          const now = moment();
          const time = moment(unixTime);
          return now.diff(time, 'hours') < 24
            ? time.format('HH:mm')
            : time.format('YYYY-MM-DD HH:mm');
        }}
        name="Time"
        scale="time"
      />
      <YAxis
        dataKey="_duration"
        name="Duration (s)"
        tickFormatter={s => s > 90 ? `${Math.round(s / 60)}m` : `${s}s`}
      />
      <Tooltip
        content={({ payload, label }) => {
          if (payload && payload.length) {
            const data: FlowRun & { time: number, _duration: number } = payload[0].payload;
            const flow = flows.find(f => f.id === data.flowID);
            return (
              <Card className="p-2 text-xs leading-normal">
                <p><strong>Agent:</strong> {flow ? flow.name : 'Unknown'}</p>
                <p>
                  <strong>Status:</strong>&nbsp;
                  <FlowRunStatusBadge status={data.status} className="px-1.5 py-0" />
                </p>
                <p><strong>Started:</strong> {moment(data.startTime).format('YYYY-MM-DD HH:mm:ss')}</p>
                <p><strong>Duration / run time:</strong> {
                  formatDuration(data.duration)} / {formatDuration(data.totalRunTime)
                }</p>
              </Card>
            );
          }
          return null;
        }}
      />
      {flows.map((flow) => (
        <Scatter
          key={flow.id}
          data={flowRuns.filter(fr => fr.flowID == flow.id).map(fr => ({
            ...fr,
            time: fr.startTime + (fr.totalRunTime * 1000),
            _duration: fr.totalRunTime,
          }))}
          name={flow.name}
          fill={`hsl(${hashString(flow.id) * 137.5 % 360}, 70%, 50%)`}
        />
      ))}
      {flowRuns.map((run) => (
        <Line
          key={run.id}
          type="linear"
          dataKey="_duration"
          data={[
            { ...run, time: run.startTime, _duration: 0 },
            { ...run, time: run.endTime, _duration: run.totalRunTime }
          ]}
          stroke={`hsl(${hashString(run.flowID) * 137.5 % 360}, 70%, 50%)`}
          strokeWidth={2}
          dot={false}
          legendType="none"
        />
      ))}
      <Legend
        content={<ScrollableLegend />}
        wrapperStyle={{
          bottom: 0,
          left: 0,
          right: 0,
          width: "100%",
          display: "flex",
          justifyContent: "center",
        }}
      />
    </ComposedChart>
  </ResponsiveContainer>
);

const ScrollableLegend: React.FC<DefaultLegendContentProps & { className?: string }> = (
  { payload, className }
) => {
  return (
    <div
      className={cn(
        "whitespace-nowrap px-4 text-sm overflow-x-auto space-x-3",
        className,
      )}
      style={{ scrollbarWidth: "none" }}
    >
      {payload.map((entry, index) => {
        if (entry.type == "none") return;
        return (
          <span key={`item-${index}`} className="inline-flex items-center">
            <span
              className="size-2.5 inline-block mr-1 rounded-full"
              style={{backgroundColor: entry.color}}
            />
            <span>{entry.value}</span>
          </span>
        )
      })}
    </div>
  );
};

function formatDuration(seconds: number): string {
  return (
    seconds < 100
      ? seconds.toPrecision(2)
      : Math.round(seconds)
  ).toString() + "s";
}

export default Monitor;

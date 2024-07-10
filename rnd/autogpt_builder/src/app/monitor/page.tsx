"use client";
import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import moment from 'moment';
import { ComposedChart, Legend, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts';
import { Pencil2Icon } from '@radix-ui/react-icons';
import AutoGPTServerAPI, { Flow, NodeExecutionResult } from '@/lib/autogpt_server_api';
import { hashString } from '@/lib/utils';
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

  const api = new AutoGPTServerAPI();

  useEffect(() => fetchFlowsAndRuns(), []);

  function fetchFlowsAndRuns() {
    // Fetch flow IDs
    api.listFlowIDs()
    .then(flowIDs => {
      Promise.all(flowIDs.map(flowID => {
        // Fetch flow run IDs
        api.listFlowRunIDs(flowID)
        .then(runIDs => {
          runIDs.map(runID => {
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
          });
        });

        // Fetch flow
        return api.getFlow(flowID);
      }))
      .then(flows => setFlows(flows));
    });
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 xl:grid-cols-10 gap-4">
      <div className="lg:col-span-2 xl:col-span-2">
        <AgentFlowList
          flows={flows}
          flowRuns={flowRuns}
          selectedFlow={selectedFlow}
          onSelectFlow={f => setSelectedFlow(f.id == selectedFlow?.id ? null : f)}
        />
      </div>
      <div className="lg:col-span-2 xl:col-span-2 space-y-4">
        <FlowRunsList
          flows={flows}
          runs={
            (
              selectedFlow
                ? flowRuns.filter(v => v.flowID == selectedFlow.id)
                : flowRuns
            )
            .toSorted((a, b) => Number(a.startTime) - Number(b.startTime))
          }
        />
      </div>
      <div className="col-span-1 lg:col-span-4 xl:col-span-6">
        {selectedFlow && (
          <Card>
            <CardHeader className="flex-row items-center justify-between space-y-0 space-x-3">
              <div>
                <CardTitle>{selectedFlow.name}</CardTitle>
                <p className="mt-2"><code>{selectedFlow.id}</code></p>
              </div>
              <Link className={buttonVariants({ variant: "outline" })} href={`/build?flowID=${selectedFlow.id}`}>
                <Pencil2Icon className="mr-2" /> Edit Flow
              </Link>
            </CardHeader>
            <CardContent>
              <FlowRunsStats
                flows={flows}
                flowRuns={flowRuns.filter(v => v.flowID == selectedFlow.id)}
              />
            </CardContent>
          </Card>
        ) || (
          <FlowRunsStats flows={flows} flowRuns={flowRuns} />
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
  duration: number // seconds

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

  // Determine aggregate startTime and duration
  const startTime = Math.min(
    ...nodeExecutionResults.map(ner => ner.start_time?.getTime() || Date.now())
  );
  const endTime = (
    ['success', 'failed'].includes(status)
      ? Math.max(...nodeExecutionResults.map(ner => ner.end_time?.getTime() || 0))
      : Date.now()
  );
  const duration = (endTime - startTime) / 1000; // Convert to seconds

  return {
    id: runID,
    flowID: flowID,
    status,
    startTime,
    duration,
    nodeExecutionResults: nodeExecutionResults
  };
}

const AgentFlowList = (
  { flows, flowRuns, selectedFlow, onSelectFlow }: {
    flows: Flow[],
    flowRuns?: FlowRun[],
    selectedFlow: Flow | null,
    onSelectFlow: (f: Flow) => void,
  }
) => (
  <Card>
    <CardHeader>
      <CardTitle>Agent Flows</CardTitle>
    </CardHeader>
    <CardContent>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            {/* <TableHead>Status</TableHead> */}
            {/* <TableHead>Last updated</TableHead> */}
            {flowRuns && <TableHead># of runs</TableHead>}
            {flowRuns && <TableHead>Last run</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {flows.map((flow) => {
            let runCount, lastRun: FlowRun | null;
            if (flowRuns) {
              const _flowRuns = flowRuns.filter(r => r.flowID == flow.id);
              runCount = _flowRuns.length;
              lastRun = runCount == 0 ? null : _flowRuns.reduce(
                (a, c) => a.startTime < c.startTime ? a : c
              );
            }
            return (
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
                {flowRuns && <TableCell>{runCount}</TableCell>}
                {flowRuns && (!lastRun ? <TableCell /> :
                <TableCell title={moment(lastRun.startTime).toString()}>
                  {moment(lastRun.startTime).fromNow()}
                </TableCell>)}
              </TableRow>
            )
          })}
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

const FlowRunsList = ({ flows, runs }: { flows: Flow[], runs: FlowRun[] }) => (
  <Card>
    <CardHeader>
      <CardTitle>Flow Runs</CardTitle>
    </CardHeader>
    <CardContent>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Flow</TableHead>
            <TableHead>Started</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Duration</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.id}>
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

const FlowRunStatusBadge = ({ status }: { status: FlowRun['status'] }) => (
  <Badge
    variant="default"
    className={
      status === 'running' ? 'bg-blue-500 dark:bg-blue-700' :
      status === 'waiting' ? 'bg-yellow-500 dark:bg-yellow-600' :
      status === 'success' ? 'bg-green-500 dark:bg-green-600' :
      'bg-red-500 dark:bg-red-700'
    }
  >
    {status}
  </Badge>
);

const ScrollableLegend = ({ payload }) => {
  return (
    <div style={{
      overflowX: 'auto',
      overflowY: 'hidden',
      whiteSpace: 'nowrap',
      padding: '10px 0',
      fontSize: '0.75em'
    }}>
      {payload.map((entry, index) => (
        <span key={`item-${index}`} style={{ display: 'inline-block', marginRight: '10px' }}>
          <span
            style={{
              display: 'inline-block',
              marginRight: '5px',
              width: '8px',
              height: '8px',
              backgroundColor: entry.color,
            }}
          />
          <span>{entry.value}</span>
        </span>
      ))}
    </div>
  );
};


const FlowRunsStats = (
  { flows, flowRuns }: {
    flows: Flow[],
    flowRuns: FlowRun[],
  }
) => {
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
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Flow Run Stats</CardTitle>
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
      </CardHeader>
      <CardContent>
        <FlowRunsTimeline flows={flows} flowRuns={flowRuns} dataMin={statsSince} className={"mb-6"} />
        <Card className="p-3">
          <p><strong>Total runs:</strong> {filteredFlowRuns.length}</p>
          <p>
            <strong>Total duration:</strong> {
              filteredFlowRuns.reduce((total, run) => total + run.duration, 0)
            } seconds
          </p>
          {/* <p><strong>Total cost:</strong> €1,23</p> */}
        </Card>
      </CardContent>
    </Card>
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
              <Card className="p-3">
                <p><strong>Flow:</strong> {flow ? flow.name : 'Unknown'}</p>
                <p><strong>Start Time:</strong> {moment(data.startTime).format('YYYY-MM-DD HH:mm:ss')}</p>
                <p>
                  <strong>Duration:</strong> {formatDuration(data.duration)}
                </p>
                <p><strong>Status:</strong> <FlowRunStatusBadge status={data.status} /></p>
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
            time: fr.startTime + (fr.duration * 1000),
            _duration: fr.duration,
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
            { ...run, time: run.startTime + (run.duration * 1000), _duration: run.duration }
          ]}
          stroke={`hsl(${hashString(run.flowID) * 137.5 % 360}, 70%, 50%)`}
          strokeWidth={2}
          dot={false}
          legendType="none"
        />
      ))}
     <Legend
        content={<ScrollableLegend />}
        wrapperStyle={{ bottom: 0, left: 0, right: 0 }}
      />
    </ComposedChart>
  </ResponsiveContainer>
);

function formatDuration(seconds: number): string {
  return (
    seconds < 100
      ? seconds.toPrecision(2)
      : Math.round(seconds)
  ).toString() + "s";
}

export default Monitor;

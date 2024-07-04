"use client";
import React, { useState } from 'react';
import moment from 'moment';
import { ComposedChart, Legend, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts';
import { Pencil2Icon } from '@radix-ui/react-icons';
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

type Flow = {
  id: number
  name: string
  status: 'active' | 'disabled' | 'failing'
  lastRun: {
    timestamp: string
    status: 'running' | 'waiting' | 'success' | 'failed'
  }
};

const FlowStatusBadge = ({ status }: { status: Flow['status'] }) => (
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
)

const AgentFlowList = (
  { flows, selectedFlow, onSelectFlow }: {
    flows: Array<Flow>,
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
            <TableHead>Status</TableHead>
            <TableHead>Most recent run</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {flows.map((flow) => (
            <TableRow
              key={flow.id}
              className="cursor-pointer"
              onClick={() => onSelectFlow(flow)}
              data-state={selectedFlow?.id == flow.id ? "selected" : null}
            >
              <TableCell>{flow.name}</TableCell>
              <TableCell><FlowStatusBadge status={flow.status} /></TableCell>
              <TableCell>
                {flow.lastRun.timestamp}
                <Badge
                  variant="default"
                  className={`ml-3 ${
                    flow.lastRun.status === 'running' ? 'bg-blue-500 dark:bg-blue-700' :
                    flow.lastRun.status === 'waiting' ? 'bg-yellow-500 dark:bg-yellow-600' :
                    flow.lastRun.status === 'success' ? 'bg-green-500 dark:bg-green-600' :
                    'bg-red-500 dark:bg-red-700'
                  }`}
                >
                  {flow.lastRun.status}
                </Badge>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </CardContent>
  </Card>
);

type FlowRun = {
  id: number
  flowId: number
  status: 'running' | 'waiting' | 'success' | 'failed'
  startTime: number // unix timestamp (ms)
  duration: number // seconds
};

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
)

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
            <TableHead>Start time</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Duration</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.id}>
              <TableCell>{flows.find(f => f.id == run.flowId)!.name}</TableCell>
              <TableCell>{moment(run.startTime).format()}</TableCell>
              <TableCell>
                <FlowRunStatusBadge status={run.status} />
              </TableCell>
              <TableCell>{run.duration}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </CardContent>
  </Card>
);


const FlowRunsStats = (
  { flows, flowRuns }: {
    flows: Flow[],
    flowRuns: FlowRun[],
  }
) => {
  const [statsSince, setStatsSince] = useState<number | "dataMin">(-24*3600)

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
        <FlowRunsTimeline flows={flows} flowRuns={flowRuns} dataMin={statsSince} />
      </CardContent>
    </Card>
  )
}

const FlowRunsTimeline = (
  { flows, flowRuns, dataMin }: {
    flows: Flow[],
    flowRuns: FlowRun[],
    dataMin: "dataMin" | number,
  }
) => (
  <ResponsiveContainer width="100%" height={120}>
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
      />
      <YAxis dataKey="_duration" name="Duration (s)" tickFormatter={s => s > 90 ? `${Math.round(s / 60)}m` : `${s}s`} />
      <Tooltip
        content={({ payload, label }) => {
          if (payload && payload.length) {
            const data = payload[0].payload;
            const flow = flows.find(f => f.id === data.flowId);
            return (
              <Card className="p-3">
                <p><strong>Flow:</strong> {flow ? flow.name : 'Unknown'}</p>
                <p><strong>Start Time:</strong> {moment(data.startTime).format('YYYY-MM-DD HH:mm:ss')}</p>
                <p>
                  <strong>Duration:</strong>
                  {
                    data.duration > 90
                      ? ` ${Math.round(data.duration/60)}m`
                      : ` ${data.duration}s`
                    }
                </p>
                <p><strong>Status:</strong> <FlowRunStatusBadge status={data.status} /></p>
              </Card>
            );
          }
          return null;
        }}
      />
      {flows.map((flow, index) => (
        <Scatter
          data={flowRuns.filter(fr => fr.flowId == flow.id).map(fr => ({
            ...fr,
            time: fr.startTime + (fr.duration * 1000),
            _duration: fr.duration,
          }))}
          name={flow.name}
          fill={`hsl(${flow.id * 137.5 % 360}, 70%, 50%)`}
        />
      ))}
      {flowRuns.map((run, index) => (
        <Line
          type="linear"
          dataKey="_duration"
          data={[
            { ...run, time: run.startTime, _duration: 0 },
            { ...run, time: run.startTime + (run.duration * 1000), _duration: run.duration }
          ]}
          stroke={`hsl(${run.flowId * 137.5 % 360}, 70%, 50%)`}
          strokeWidth={2}
          dot={false}
          legendType="none"
        />
      ))}
      <Legend />
    </ComposedChart>
  </ResponsiveContainer>
);

const Monitor = () => {
  const [selectedFlow, setSelectedFlow] = useState<Flow | null>(null);

  // Mock data
  const flows: Array<Flow> = [
    { id: 1, name: 'JARVIS', status: 'active', lastRun: { timestamp: '5 minutes ago', status: 'waiting' } },
    { id: 2, name: 'Time machine', status: 'failing', lastRun: { timestamp: '10 minutes ago', status: 'failed' } },
    { id: 3, name: 'BlueSky digest', status: 'active', lastRun: { timestamp: '2 minutes ago', status: 'success' } },
  ];

  const flowRuns: Array<FlowRun> = [
    { flowId: 1, id: 1, startTime: Date.now() - (4 * 3.6e6 + 55 * 60e3), status: 'success', duration: 59 },
    { flowId: 1, id: 3, startTime: Date.now() - (3 * 3.6e6 + 23 * 60e3), status: 'success', duration: 102 },
    { flowId: 1, id: 6, startTime: Date.now() - (2 * 3.6e6 + 8 * 60e3), status: 'waiting', duration: 133 },
    { flowId: 2, id: 2, startTime: Date.now() - (4 * 3.6e6 + 11 * 60e3), status: 'failed', duration: 151 },
    { flowId: 2, id: 5, startTime: Date.now() - (2 * 3.6e6 + 57 * 60e3), status: 'failed', duration: 77 },
    { flowId: 2, id: 8, startTime: Date.now() - (1 * 3.6e6 + 42 * 60e3), status: 'failed', duration: 185 },
    { flowId: 3, id: 4, startTime: Date.now() - (3 * 3.6e6 + 15 * 60e3), status: 'success', duration: 67 },
    { flowId: 3, id: 7, startTime: Date.now() - (1 * 3.6e6 + 49 * 60e3), status: 'success', duration: 83 },
    { flowId: 3, id: 9, startTime: Date.now() - (31 * 60e3), status: 'success', duration: 118 },
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 xl:grid-cols-10 gap-4">
      <div className="lg:col-span-2 xl:col-span-2">
        <AgentFlowList
          flows={flows}
          selectedFlow={selectedFlow}
          onSelectFlow={(f: Flow) => setSelectedFlow(f.id == selectedFlow?.id ? null : f)}
        />
      </div>
      <div className="lg:col-span-2 xl:col-span-2 space-y-4">
        <FlowRunsList
          flows={flows}
          runs={
            (
              selectedFlow
                ? flowRuns.filter(v => v.flowId == selectedFlow.id)
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
                <CardTitle>{selectedFlow.name}</CardTitle>
                <Button variant="outline">
                  <Pencil2Icon className="mr-2" /> Edit Flow
                </Button>
              </CardHeader>
              <CardContent>
                <FlowRunsStats
                  flows={flows}
                  flowRuns={flowRuns.filter(v => v.flowId == selectedFlow.id)}
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

export default Monitor;

"use client";
import React, { useEffect, useState } from "react";
import Link from "next/link";
import moment from "moment";
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
} from "recharts";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import AutoGPTServerAPI, {
  Graph,
  GraphMeta,
  NodeExecutionResult,
  safeCopyGraph,
} from "@/lib/autogpt-server-api";
import {
  ChevronDownIcon,
  ClockIcon,
  EnterIcon,
  ExitIcon,
  Pencil2Icon,
} from "@radix-ui/react-icons";
import { cn, exportAsJSONFile, hashString } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { AgentImportForm } from "@/components/agent-import-form";

const Monitor = () => {
  const [flows, setFlows] = useState<GraphMeta[]>([]);
  const [flowRuns, setFlowRuns] = useState<FlowRun[]>([]);
  const [selectedFlow, setSelectedFlow] = useState<GraphMeta | null>(null);
  const [selectedRun, setSelectedRun] = useState<FlowRun | null>(null);

  const api = new AutoGPTServerAPI();

  useEffect(() => fetchFlowsAndRuns(), []);
  useEffect(() => {
    const intervalId = setInterval(
      () => flows.map((f) => refreshFlowRuns(f.id)),
      5000,
    );
    return () => clearInterval(intervalId);
  }, []);

  function fetchFlowsAndRuns() {
    api.listGraphs().then((flows) => {
      setFlows(flows);
      flows.map((flow) => refreshFlowRuns(flow.id));
    });
  }

  function refreshFlowRuns(flowID: string) {
    // Fetch flow run IDs
    api.listGraphRunIDs(flowID).then((runIDs) =>
      runIDs.map((runID) => {
        let run;
        if (
          (run = flowRuns.find((fr) => fr.id == runID)) &&
          !["waiting", "running"].includes(run.status)
        ) {
          return;
        }

        // Fetch flow run
        api.getGraphExecutionInfo(flowID, runID).then((execInfo) =>
          setFlowRuns((flowRuns) => {
            if (execInfo.length == 0) return flowRuns;

            const flowRunIndex = flowRuns.findIndex((fr) => fr.id == runID);
            const flowRun = flowRunFromNodeExecutionResults(execInfo);
            if (flowRunIndex > -1) {
              flowRuns.splice(flowRunIndex, 1, flowRun);
            } else {
              flowRuns.push(flowRun);
            }
            return [...flowRuns];
          }),
        );
      }),
    );
  }

  const column1 = "md:col-span-2 xl:col-span-3 xxl:col-span-2";
  const column2 = "md:col-span-3 lg:col-span-2 xl:col-span-3 space-y-4";
  const column3 = "col-span-full xl:col-span-4 xxl:col-span-5";

  return (
    <div className="grid grid-cols-1 md:grid-cols-5 lg:grid-cols-4 xl:grid-cols-10 gap-4">
      <AgentFlowList
        className={column1}
        flows={flows}
        flowRuns={flowRuns}
        selectedFlow={selectedFlow}
        onSelectFlow={(f) => {
          setSelectedRun(null);
          setSelectedFlow(f.id == selectedFlow?.id ? null : f);
        }}
      />
      <FlowRunsList
        className={column2}
        flows={flows}
        runs={(selectedFlow
          ? flowRuns.filter((v) => v.graphID == selectedFlow.id)
          : flowRuns
        ).toSorted((a, b) => Number(a.startTime) - Number(b.startTime))}
        selectedRun={selectedRun}
        onSelectRun={(r) => setSelectedRun(r.id == selectedRun?.id ? null : r)}
      />
      {(selectedRun && (
        <FlowRunInfo
          flow={selectedFlow || flows.find((f) => f.id == selectedRun.graphID)!}
          flowRun={selectedRun}
          className={column3}
        />
      )) ||
        (selectedFlow && (
          <FlowInfo
            flow={selectedFlow}
            flowRuns={flowRuns.filter((r) => r.graphID == selectedFlow.id)}
            className={column3}
          />
        )) || (
          <Card className={`p-6 ${column3}`}>
            <FlowRunsStats flows={flows} flowRuns={flowRuns} />
          </Card>
        )}
    </div>
  );
};

type FlowRun = {
  id: string;
  graphID: string;
  graphVersion: number;
  status: "running" | "waiting" | "success" | "failed";
  startTime: number; // unix timestamp (ms)
  endTime: number; // unix timestamp (ms)
  duration: number; // seconds
  totalRunTime: number; // seconds

  nodeExecutionResults: NodeExecutionResult[];
};

function flowRunFromNodeExecutionResults(
  nodeExecutionResults: NodeExecutionResult[],
): FlowRun {
  // Determine overall status
  let status: "running" | "waiting" | "success" | "failed" = "success";
  for (const execution of nodeExecutionResults) {
    if (execution.status === "FAILED") {
      status = "failed";
      break;
    } else if (["QUEUED", "RUNNING"].includes(execution.status)) {
      status = "running";
      break;
    } else if (execution.status === "INCOMPLETE") {
      status = "waiting";
    }
  }

  // Determine aggregate startTime, endTime, and totalRunTime
  const now = Date.now();
  const startTime = Math.min(
    ...nodeExecutionResults.map((ner) => ner.add_time.getTime()),
    now,
  );
  const endTime = ["success", "failed"].includes(status)
    ? Math.max(
        ...nodeExecutionResults.map((ner) => ner.end_time?.getTime() || 0),
        startTime,
      )
    : now;
  const duration = (endTime - startTime) / 1000; // Convert to seconds
  const totalRunTime =
    nodeExecutionResults.reduce(
      (cum, node) =>
        cum +
        ((node.end_time?.getTime() ?? now) -
          (node.start_time?.getTime() ?? now)),
      0,
    ) / 1000;

  return {
    id: nodeExecutionResults[0].graph_exec_id,
    graphID: nodeExecutionResults[0].graph_id,
    graphVersion: nodeExecutionResults[0].graph_version,
    status,
    startTime,
    endTime,
    duration,
    totalRunTime,
    nodeExecutionResults: nodeExecutionResults,
  };
}

const AgentFlowList = ({
  flows,
  flowRuns,
  selectedFlow,
  onSelectFlow,
  className,
}: {
  flows: GraphMeta[];
  flowRuns?: FlowRun[];
  selectedFlow: GraphMeta | null;
  onSelectFlow: (f: GraphMeta) => void;
  className?: string;
}) => {
  const [templates, setTemplates] = useState<GraphMeta[]>([]);
  const api = new AutoGPTServerAPI();
  useEffect(() => {
    api.listTemplates().then((templates) => setTemplates(templates));
  }, []);

  return (
    <Card className={className}>
      <CardHeader className="flex-row justify-between items-center space-x-3 space-y-0">
        <CardTitle>Agents</CardTitle>

        <div className="flex items-center">
          {/* Split "Create" button */}
          <Button variant="outline" className="rounded-r-none" asChild>
            <Link href="/build">Create</Link>
          </Button>
          <Dialog>
            {/* https://ui.shadcn.com/docs/components/dialog#notes */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  className={"rounded-l-none border-l-0 px-2"}
                >
                  <ChevronDownIcon />
                </Button>
              </DropdownMenuTrigger>

              <DropdownMenuContent>
                <DialogTrigger asChild>
                  <DropdownMenuItem>
                    <EnterIcon className="mr-2" /> Import from file
                  </DropdownMenuItem>
                </DialogTrigger>
                {templates.length > 0 && (
                  <>
                    {/* List of templates */}
                    <DropdownMenuSeparator />
                    <DropdownMenuLabel>Use a template</DropdownMenuLabel>
                    {templates.map((template) => (
                      <DropdownMenuItem
                        key={template.id}
                        onClick={() => {
                          api
                            .createGraph(template.id, template.version)
                            .then((newGraph) => {
                              window.location.href = `/build?flowID=${newGraph.id}`;
                            });
                        }}
                      >
                        {template.name}
                      </DropdownMenuItem>
                    ))}
                  </>
                )}
              </DropdownMenuContent>
            </DropdownMenu>

            <DialogContent>
              <DialogHeader className="text-lg">
                Import an Agent (template) from a file
              </DialogHeader>
              <AgentImportForm />
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>

      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              {/* <TableHead>Status</TableHead> */}
              {/* <TableHead>Last updated</TableHead> */}
              {flowRuns && (
                <TableHead className="md:hidden lg:table-cell">
                  # of runs
                </TableHead>
              )}
              {flowRuns && <TableHead>Last run</TableHead>}
            </TableRow>
          </TableHeader>
          <TableBody>
            {flows
              .map((flow) => {
                let runCount = 0,
                  lastRun: FlowRun | null = null;
                if (flowRuns) {
                  const _flowRuns = flowRuns.filter(
                    (r) => r.graphID == flow.id,
                  );
                  runCount = _flowRuns.length;
                  lastRun =
                    runCount == 0
                      ? null
                      : _flowRuns.reduce((a, c) =>
                          a.startTime > c.startTime ? a : c,
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
                  {flowRuns && (
                    <TableCell className="md:hidden lg:table-cell">
                      {runCount}
                    </TableCell>
                  )}
                  {flowRuns &&
                    (!lastRun ? (
                      <TableCell />
                    ) : (
                      <TableCell title={moment(lastRun.startTime).toString()}>
                        {moment(lastRun.startTime).fromNow()}
                      </TableCell>
                    ))}
                </TableRow>
              ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
};

const FlowStatusBadge = ({
  status,
}: {
  status: "active" | "disabled" | "failing";
}) => (
  <Badge
    variant="default"
    className={
      status === "active"
        ? "bg-green-500 dark:bg-green-600"
        : status === "failing"
          ? "bg-red-500 dark:bg-red-700"
          : "bg-gray-500 dark:bg-gray-600"
    }
  >
    {status}
  </Badge>
);

const FlowRunsList: React.FC<{
  flows: GraphMeta[];
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
              <TableCell>
                {flows.find((f) => f.id == run.graphID)!.name}
              </TableCell>
              <TableCell>{moment(run.startTime).format("HH:mm")}</TableCell>
              <TableCell>
                <FlowRunStatusBadge status={run.status} />
              </TableCell>
              <TableCell>{formatDuration(run.duration)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </CardContent>
  </Card>
);

const FlowRunStatusBadge: React.FC<{
  status: FlowRun["status"];
  className?: string;
}> = ({ status, className }) => (
  <Badge
    variant="default"
    className={cn(
      status === "running"
        ? "bg-blue-500 dark:bg-blue-700"
        : status === "waiting"
          ? "bg-yellow-500 dark:bg-yellow-600"
          : status === "success"
            ? "bg-green-500 dark:bg-green-600"
            : "bg-red-500 dark:bg-red-700",
      className,
    )}
  >
    {status}
  </Badge>
);

const FlowInfo: React.FC<
  React.HTMLAttributes<HTMLDivElement> & {
    flow: GraphMeta;
    flowRuns: FlowRun[];
    flowVersion?: number | "all";
  }
> = ({ flow, flowRuns, flowVersion, ...props }) => {
  const api = new AutoGPTServerAPI();

  const [flowVersions, setFlowVersions] = useState<Graph[] | null>(null);
  const [selectedVersion, setSelectedFlowVersion] = useState(
    flowVersion ?? "all",
  );
  const selectedFlowVersion: Graph | undefined = flowVersions?.find(
    (v) =>
      v.version == (selectedVersion == "all" ? flow.version : selectedVersion),
  );

  useEffect(() => {
    api.getGraphAllVersions(flow.id).then((result) => setFlowVersions(result));
  }, [flow.id]);

  return (
    <Card {...props}>
      <CardHeader className="flex-row justify-between space-y-0 space-x-3">
        <div>
          <CardTitle>
            {flow.name} <span className="font-light">v{flow.version}</span>
          </CardTitle>
          <p className="mt-2">
            Agent ID: <code>{flow.id}</code>
          </p>
        </div>
        <div className="flex items-start space-x-2">
          {(flowVersions?.length ?? 0) > 1 && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline">
                  <ClockIcon className="mr-2" />
                  {selectedVersion == "all"
                    ? "All versions"
                    : `Version ${selectedVersion}`}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>Choose a version</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuRadioGroup
                  value={String(selectedVersion)}
                  onValueChange={(choice) =>
                    setSelectedFlowVersion(
                      choice == "all" ? choice : Number(choice),
                    )
                  }
                >
                  <DropdownMenuRadioItem value="all">
                    All versions
                  </DropdownMenuRadioItem>
                  {flowVersions?.map((v) => (
                    <DropdownMenuRadioItem
                      key={v.version}
                      value={v.version.toString()}
                    >
                      Version {v.version}
                      {v.is_active ? " (active)" : ""}
                    </DropdownMenuRadioItem>
                  ))}
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
          <Link
            className={buttonVariants({ variant: "outline" })}
            href={`/build?flowID=${flow.id}`}
          >
            <Pencil2Icon className="mr-2" /> Edit
          </Link>
          <Button
            variant="outline"
            className="px-2.5"
            title="Export to a JSON-file"
            onClick={async () =>
              exportAsJSONFile(
                safeCopyGraph(
                  flowVersions!.find(
                    (v) => v.version == selectedFlowVersion!.version,
                  )!,
                  await api.getBlocks(),
                ),
                `${flow.name}_v${selectedFlowVersion!.version}.json`,
              )
            }
          >
            <ExitIcon />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <FlowRunsStats
          flows={[selectedFlowVersion ?? flow]}
          flowRuns={flowRuns.filter(
            (r) =>
              r.graphID == flow.id &&
              (selectedVersion == "all" || r.graphVersion == selectedVersion),
          )}
        />
      </CardContent>
    </Card>
  );
};

const FlowRunInfo: React.FC<
  React.HTMLAttributes<HTMLDivElement> & {
    flow: GraphMeta;
    flowRun: FlowRun;
  }
> = ({ flow, flowRun, ...props }) => {
  if (flowRun.graphID != flow.id) {
    throw new Error(
      `FlowRunInfo can't be used with non-matching flowRun.flowID and flow.id`,
    );
  }

  return (
    <Card {...props}>
      <CardHeader className="flex-row items-center justify-between space-y-0 space-x-3">
        <div>
          <CardTitle>
            {flow.name} <span className="font-light">v{flow.version}</span>
          </CardTitle>
          <p className="mt-2">
            Agent ID: <code>{flow.id}</code>
          </p>
          <p className="mt-1">
            Run ID: <code>{flowRun.id}</code>
          </p>
        </div>
        <Link
          className={buttonVariants({ variant: "outline" })}
          href={`/build?flowID=${flow.id}`}
        >
          <Pencil2Icon className="mr-2" /> Edit Agent
        </Link>
      </CardHeader>
      <CardContent>
        <p>
          <strong>Status:</strong>{" "}
          <FlowRunStatusBadge status={flowRun.status} />
        </p>
        <p>
          <strong>Started:</strong>{" "}
          {moment(flowRun.startTime).format("YYYY-MM-DD HH:mm:ss")}
        </p>
        <p>
          <strong>Finished:</strong>{" "}
          {moment(flowRun.endTime).format("YYYY-MM-DD HH:mm:ss")}
        </p>
        <p>
          <strong>Duration (run time):</strong> {flowRun.duration} (
          {flowRun.totalRunTime}) seconds
        </p>
        {/* <p><strong>Total cost:</strong> €1,23</p> */}
      </CardContent>
    </Card>
  );
};

const FlowRunsStats: React.FC<{
  flows: GraphMeta[];
  flowRuns: FlowRun[];
  title?: string;
  className?: string;
}> = ({ flows, flowRuns, title, className }) => {
  /* "dateMin": since the first flow in the dataset
   * number > 0: custom date (unix timestamp)
   * number < 0: offset relative to Date.now() (in seconds) */
  const [statsSince, setStatsSince] = useState<number | "dataMin">(-24 * 3600);
  const statsSinceTimestamp = // unix timestamp or null
    typeof statsSince == "string"
      ? null
      : statsSince < 0
        ? Date.now() + statsSince * 1000
        : statsSince;
  const filteredFlowRuns =
    statsSinceTimestamp != null
      ? flowRuns.filter((fr) => fr.startTime > statsSinceTimestamp)
      : flowRuns;

  return (
    <div className={className}>
      <div className="flex flex-row items-center justify-between">
        <CardTitle>{title || "Stats"}</CardTitle>
        <div className="flex space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setStatsSince(-2 * 3600)}
          >
            2h
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setStatsSince(-8 * 3600)}
          >
            8h
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setStatsSince(-24 * 3600)}
          >
            24h
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setStatsSince(-7 * 24 * 3600)}
          >
            7d
          </Button>
          <Popover>
            <PopoverTrigger asChild>
              <Button variant={"outline"} size="sm">
                Custom
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                onSelect={(_, selectedDay) =>
                  setStatsSince(selectedDay.getTime())
                }
                initialFocus
              />
            </PopoverContent>
          </Popover>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setStatsSince("dataMin")}
          >
            All
          </Button>
        </div>
      </div>
      <FlowRunsTimeline
        flows={flows}
        flowRuns={flowRuns}
        dataMin={statsSince}
        className="mt-3"
      />
      <hr className="my-4" />
      <div>
        <p>
          <strong>Total runs:</strong> {filteredFlowRuns.length}
        </p>
        <p>
          <strong>Total run time:</strong>{" "}
          {filteredFlowRuns.reduce((total, run) => total + run.totalRunTime, 0)}{" "}
          seconds
        </p>
        {/* <p><strong>Total cost:</strong> €1,23</p> */}
      </div>
    </div>
  );
};

const FlowRunsTimeline = ({
  flows,
  flowRuns,
  dataMin,
  className,
}: {
  flows: GraphMeta[];
  flowRuns: FlowRun[];
  dataMin: "dataMin" | number;
  className?: string;
}) => (
  /* TODO: make logarithmic? */
  <ResponsiveContainer width="100%" height={120} className={className}>
    <ComposedChart>
      <XAxis
        dataKey="time"
        type="number"
        domain={[
          typeof dataMin == "string"
            ? dataMin
            : dataMin < 0
              ? Date.now() + dataMin * 1000
              : dataMin,
          Date.now(),
        ]}
        allowDataOverflow={true}
        tickFormatter={(unixTime) => {
          const now = moment();
          const time = moment(unixTime);
          return now.diff(time, "hours") < 24
            ? time.format("HH:mm")
            : time.format("YYYY-MM-DD HH:mm");
        }}
        name="Time"
        scale="time"
      />
      <YAxis
        dataKey="_duration"
        name="Duration (s)"
        tickFormatter={(s) => (s > 90 ? `${Math.round(s / 60)}m` : `${s}s`)}
      />
      <Tooltip
        content={({ payload, label }) => {
          if (payload && payload.length) {
            const data: FlowRun & { time: number; _duration: number } =
              payload[0].payload;
            const flow = flows.find((f) => f.id === data.graphID);
            return (
              <Card className="p-2 text-xs leading-normal">
                <p>
                  <strong>Agent:</strong> {flow ? flow.name : "Unknown"}
                </p>
                <p>
                  <strong>Status:</strong>&nbsp;
                  <FlowRunStatusBadge
                    status={data.status}
                    className="px-1.5 py-0"
                  />
                </p>
                <p>
                  <strong>Started:</strong>{" "}
                  {moment(data.startTime).format("YYYY-MM-DD HH:mm:ss")}
                </p>
                <p>
                  <strong>Duration / run time:</strong>{" "}
                  {formatDuration(data.duration)} /{" "}
                  {formatDuration(data.totalRunTime)}
                </p>
              </Card>
            );
          }
          return null;
        }}
      />
      {flows.map((flow) => (
        <Scatter
          key={flow.id}
          data={flowRuns
            .filter((fr) => fr.graphID == flow.id)
            .map((fr) => ({
              ...fr,
              time: fr.startTime + fr.totalRunTime * 1000,
              _duration: fr.totalRunTime,
            }))}
          name={flow.name}
          fill={`hsl(${(hashString(flow.id) * 137.5) % 360}, 70%, 50%)`}
        />
      ))}
      {flowRuns.map((run) => (
        <Line
          key={run.id}
          type="linear"
          dataKey="_duration"
          data={[
            { ...run, time: run.startTime, _duration: 0 },
            { ...run, time: run.endTime, _duration: run.totalRunTime },
          ]}
          stroke={`hsl(${(hashString(run.graphID) * 137.5) % 360}, 70%, 50%)`}
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

const ScrollableLegend: React.FC<
  DefaultLegendContentProps & { className?: string }
> = ({ payload, className }) => {
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
              style={{ backgroundColor: entry.color }}
            />
            <span>{entry.value}</span>
          </span>
        );
      })}
    </div>
  );
};

function formatDuration(seconds: number): string {
  return (
    (seconds < 100 ? seconds.toPrecision(2) : Math.round(seconds)).toString() +
    "s"
  );
}

export default Monitor;

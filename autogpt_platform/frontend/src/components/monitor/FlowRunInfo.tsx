import React, { useCallback, useEffect, useState } from "react";
import {
  GraphExecutionMeta,
  LibraryAgent,
  NodeExecutionResult,
  SpecialBlockID,
} from "@/lib/autogpt-server-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Link from "next/link";
import { Button, buttonVariants } from "@/components/ui/button";
import { IconSquare } from "@/components/ui/icons";
import { ExitIcon, Pencil2Icon } from "@radix-ui/react-icons";
import moment from "moment/moment";
import { FlowRunStatusBadge } from "@/components/monitor/FlowRunStatusBadge";
import RunnerOutputUI, { BlockOutput } from "../runner-ui/RunnerOutputUI";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

export const FlowRunInfo: React.FC<
  React.HTMLAttributes<HTMLDivElement> & {
    agent: LibraryAgent;
    execution: GraphExecutionMeta;
  }
> = ({ agent, execution, ...props }) => {
  const [isOutputOpen, setIsOutputOpen] = useState(false);
  const [blockOutputs, setBlockOutputs] = useState<BlockOutput[]>([]);
  const api = useBackendAPI();

  const fetchBlockResults = useCallback(async () => {
    const graph = await api.getGraph(agent.graph_id, agent.graph_version);
    const graphExecution = await api.getGraphExecutionInfo(
      agent.graph_id,
      execution.id,
    );

    // Transform results to BlockOutput format
    setBlockOutputs(
      Object.entries(graphExecution.outputs).flatMap(([key, values]) =>
        values.map(
          (value) =>
            ({
              metadata: {
                name: graph.output_schema.properties[key].title || "Output",
                description:
                  graph.output_schema.properties[key].description ||
                  "Output from the agent",
              },
              result: value,
            }) satisfies BlockOutput,
        ),
      ),
    );
  }, [api, agent.graph_id, agent.graph_version, execution.id]);

  // Fetch graph and execution data
  useEffect(() => {
    if (!isOutputOpen) return;
    fetchBlockResults();
  }, [isOutputOpen, fetchBlockResults]);

  if (execution.graph_id != agent.graph_id) {
    throw new Error(
      `FlowRunInfo can't be used with non-matching execution.graph_id and flow.id`,
    );
  }

  const handleStopRun = useCallback(() => {
    api.stopGraphExecution(agent.graph_id, execution.id);
  }, [api, agent.graph_id, execution.id]);

  return (
    <>
      <Card {...props}>
        <CardHeader className="flex-row items-center justify-between space-x-3 space-y-0">
          <div>
            <CardTitle>
              {agent.name}{" "}
              <span className="font-light">v{execution.graph_version}</span>
            </CardTitle>
          </div>
          <div className="flex space-x-2">
            {execution.status === "RUNNING" && (
              <Button onClick={handleStopRun} variant="destructive">
                <IconSquare className="mr-2" /> Stop Run
              </Button>
            )}
            <Button onClick={() => setIsOutputOpen(true)} variant="outline">
              <ExitIcon className="mr-2" /> View Outputs
            </Button>
            {agent.can_access_graph && (
              <Link
                className={buttonVariants({ variant: "default" })}
                href={`/build?flowID=${execution.graph_id}&flowVersion=${execution.graph_version}&flowExecutionID=${execution.id}`}
              >
                <Pencil2Icon className="mr-2" /> Open in Builder
              </Link>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <p className="hidden">
            <strong>Agent ID:</strong> <code>{agent.graph_id}</code>
          </p>
          <p className="hidden">
            <strong>Run ID:</strong> <code>{execution.id}</code>
          </p>
          <div>
            <strong>Status:</strong>{" "}
            <FlowRunStatusBadge status={execution.status} />
          </div>
          <p>
            <strong>Started:</strong>{" "}
            {moment(execution.started_at).format("YYYY-MM-DD HH:mm:ss")}
          </p>
          <p>
            <strong>Finished:</strong>{" "}
            {moment(execution.ended_at).format("YYYY-MM-DD HH:mm:ss")}
          </p>
          {execution.stats && (
            <p>
              <strong>Duration (run time):</strong>{" "}
              {execution.stats.duration.toFixed(1)} (
              {execution.stats.node_exec_time.toFixed(1)}) seconds
            </p>
          )}
        </CardContent>
      </Card>
      <RunnerOutputUI
        isOpen={isOutputOpen}
        onClose={() => setIsOutputOpen(false)}
        blockOutputs={blockOutputs}
      />
    </>
  );
};

export default FlowRunInfo;

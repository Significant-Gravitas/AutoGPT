import React, { useCallback, useEffect, useMemo, useState } from "react";
import AutoGPTServerAPI, {
  BlockIORootSchema,
  Graph,
  GraphMeta,
  NodeExecutionResult,
} from "@/lib/autogpt-server-api";
import { FlowRun } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Link from "next/link";
import { Button, buttonVariants } from "@/components/ui/button";
import { IconSquare } from "@/components/ui/icons";
import { ExitIcon, Pencil2Icon } from "@radix-ui/react-icons";
import moment from "moment/moment";
import { FlowRunStatusBadge } from "@/components/monitor/FlowRunStatusBadge";
import RunnerOutputUI, { BlockOutput } from "../runner-ui/RunnerOutputUI";

export const FlowRunInfo: React.FC<
  React.HTMLAttributes<HTMLDivElement> & {
    flow: GraphMeta;
    flowRun: FlowRun;
  }
> = ({ flow, flowRun, ...props }) => {
  const [isOutputOpen, setIsOutputOpen] = useState(false);
  const [graph, setGraph] = useState<Graph | null>(null);
  const [executionResults, setExecutionResults] = useState<
    NodeExecutionResult[] | null
  >(null);
  const [blockOutputs, setBlockOutputs] = useState<BlockOutput[]>([]);
  const api = useMemo(() => new AutoGPTServerAPI(), []);

  // Fetch graph and execution data
  useEffect(() => {
    api.getGraph(flow.id, flow.version).then((graph) => {
      setGraph(graph);
    });
    api.getGraphExecutionInfo(flow.id, flowRun.id).then((executions) => {
      setExecutionResults(executions);
    });
  }, [api]);

  useEffect(() => {
    if (!graph || !executionResults) {
      return;
    }

    // Filter nodes to only get output blocks
    const outputNodes = graph.nodes.filter(
      (node) => node.block_id === "363ae599-353e-4804-937e-b2ee3cef3da4",
    );

    // Create a map of the latest COMPLETED execution results by node_id
    const latestCompletedResults = executionResults
      .filter((result) => result.status === "COMPLETED")
      .reduce((acc, result) => {
        const existing = acc.get(result.node_id);

        // Compare dates if there's an existing result
        if (existing) {
          const existingDate = existing.end_time || existing.add_time;
          const currentDate = result.end_time || result.add_time;

          if (currentDate > existingDate) {
            acc.set(result.node_id, result);
          }
        } else {
          acc.set(result.node_id, result);
        }

        return acc;
      }, new Map<string, NodeExecutionResult>());

    // Transform output nodes to BlockOutput format
    setBlockOutputs(
      outputNodes.map((node) => ({
        id: node.id,
        outputSchema: {} as BlockIORootSchema,
        hardcodedValues: {
          name:
            latestCompletedResults.get(node.id)?.input_data.name || "Output",
          description:
            latestCompletedResults.get(node.id)?.input_data.description ||
            "Output from the agent",
        },
        result: latestCompletedResults.get(node.id)?.output_data || undefined,
      })),
    );
  }, [graph, executionResults]);

  if (flowRun.graphID != flow.id) {
    throw new Error(
      `FlowRunInfo can't be used with non-matching flowRun.flowID and flow.id`,
    );
  }

  const handleStopRun = useCallback(() => {
    api.stopGraphExecution(flow.id, flowRun.id);
  }, [flow.id, flowRun.id]);

  return (
    <>
      <Card {...props}>
        <CardHeader className="flex-row items-center justify-between space-x-3 space-y-0">
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
          <div className="flex space-x-2">
            {flowRun.status === "running" && (
              <Button onClick={handleStopRun} variant="destructive">
                <IconSquare className="mr-2" /> Stop Run
              </Button>
            )}
            <Button onClick={() => setIsOutputOpen(true)} variant="outline">
              <ExitIcon className="mr-2" /> View Outputs
            </Button>
            <Link
              className={buttonVariants({ variant: "outline" })}
              href={`/build?flowID=${flow.id}`}
            >
              <Pencil2Icon className="mr-2" /> Edit Agent
            </Link>
          </div>
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
      <RunnerOutputUI
        isOpen={isOutputOpen}
        onClose={() => setIsOutputOpen(false)}
        blockOutputs={blockOutputs}
      />
    </>
  );
};
export default FlowRunInfo;

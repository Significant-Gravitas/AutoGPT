import React, { useCallback, useEffect, useMemo, useState } from "react";
import AutoGPTServerAPI, {
  BlockIORootSchema,
  Graph,
  GraphMeta,
  NodeExecutionResult,
  SpecialBlockID,
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
  const [blockOutputs, setBlockOutputs] = useState<BlockOutput[]>([]);
  const api = useMemo(() => new AutoGPTServerAPI(), []);

  const fetchBlockResults = useCallback(async () => {
    const executionResults = await api.getGraphExecutionInfo(
      flow.id,
      flowRun.id,
    );

    // Create a map of the latest COMPLETED execution results of output nodes by node_id
    const latestCompletedResults = executionResults
      .filter(
        (result) =>
          result.status === "COMPLETED" &&
          result.block_id === SpecialBlockID.OUTPUT,
      )
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

    // Transform results to BlockOutput format
    setBlockOutputs(
      Array.from(latestCompletedResults.values()).map((result) => ({
        id: result.node_id,
        type: "output" as const,
        hardcodedValues: {
          name: result.input_data.name || "Output",
          description: result.input_data.description || "Output from the agent",
          value: result.input_data.value,
        },
        // Change this line to extract the array directly
        result: result.output_data?.output || undefined,
      })),
    );
  }, [api, flow.id, flowRun.id]);

  // Fetch graph and execution data
  useEffect(() => {
    if (!isOutputOpen || blockOutputs.length > 0) {
      return;
    }

    fetchBlockResults();
  }, [isOutputOpen, blockOutputs, fetchBlockResults]);

  if (flowRun.graphID != flow.id) {
    throw new Error(
      `FlowRunInfo can't be used with non-matching flowRun.flowID and flow.id`,
    );
  }

  const handleStopRun = useCallback(() => {
    api.stopGraphExecution(flow.id, flowRun.id);
  }, [api, flow.id, flowRun.id]);

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
              className={buttonVariants({ variant: "default" })}
              href={`/build?flowID=${flow.id}`}
            >
              <Pencil2Icon className="mr-2" /> Open in Builder
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          <div>
            <strong>Status:</strong>{" "}
            <FlowRunStatusBadge status={flowRun.status} />
          </div>
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

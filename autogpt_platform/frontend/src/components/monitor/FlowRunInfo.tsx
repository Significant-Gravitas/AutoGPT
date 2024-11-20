import React, { useCallback } from "react";
import AutoGPTServerAPI, { GraphMeta } from "@/lib/autogpt-server-api";
import { FlowRun } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Link from "next/link";
import { Button, buttonVariants } from "@/components/ui/button";
import { IconSquare } from "@/components/ui/icons";
import { Pencil2Icon } from "@radix-ui/react-icons";
import moment from "moment/moment";
import { FlowRunStatusBadge } from "@/components/monitor/FlowRunStatusBadge";

export const FlowRunInfo: React.FC<
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

  const handleStopRun = useCallback(() => {
    const api = new AutoGPTServerAPI();
    api.stopGraphExecution(flow.id, flowRun.id);
  }, [flow.id, flowRun.id]);

  return (
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
        {/* <p><strong>Total cost:</strong> €1,23</p> */}
      </CardContent>
    </Card>
  );
};
export default FlowRunInfo;

import React from "react";
import { GraphMeta } from "@/lib/autogpt-server-api";
import { FlowRun } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import moment from "moment/moment";
import { FlowRunStatusBadge } from "@/components/monitor/FlowRunStatusBadge";
import { TextRenderer } from "../ui/render";

export const FlowRunsList: React.FC<{
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
                <TextRenderer
                  value={flows.find((f) => f.id == run.graphID)!.name}
                  truncateLengthLimit={30}
                />
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

function formatDuration(seconds: number): string {
  return (
    (seconds < 100 ? seconds.toPrecision(2) : Math.round(seconds)).toString() +
    "s"
  );
}

export default FlowRunsList;

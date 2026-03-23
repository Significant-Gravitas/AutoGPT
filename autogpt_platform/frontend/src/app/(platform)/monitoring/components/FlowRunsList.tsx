import React from "react";
import { GraphExecutionMeta, LibraryAgent } from "@/lib/autogpt-server-api";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/__legacy__/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/__legacy__/ui/table";
import moment from "moment/moment";
import { FlowRunStatusBadge } from "@/app/(platform)/monitoring/components/FlowRunStatusBadge";
import { TextRenderer } from "../../../../components/__legacy__/ui/render";
import { Play } from "@phosphor-icons/react";

interface Props {
  flows: LibraryAgent[];
  executions: GraphExecutionMeta[];
  className?: string;
  selectedRun?: GraphExecutionMeta | null;
  onSelectRun: (r: GraphExecutionMeta) => void;
  isLoading?: boolean;
}

export function FlowRunsList({
  flows,
  executions,
  selectedRun,
  onSelectRun,
  className,
  isLoading,
}: Props) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Runs</CardTitle>
      </CardHeader>
      <CardContent>
        {!isLoading && executions.length === 0 && (
          <div className="flex flex-col items-center gap-3 py-10 text-center text-muted-foreground">
            <Play size={40} weight="thin" />
            <p className="text-sm font-medium">No runs yet</p>
            <p className="max-w-[200px] text-xs">
              Select an agent and run it to see execution history here.
            </p>
          </div>
        )}

        {(isLoading || executions.length > 0) && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Agent</TableHead>
                <TableHead>Started</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Duration</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody data-testid="flow-runs-list-body">
              {executions.map((execution) => (
                <TableRow
                  key={execution.id}
                  data-testid={`flow-run-${execution.id}-graph-${execution.graph_id}`}
                  data-runid={execution.id}
                  data-graphid={execution.graph_id}
                  className="cursor-pointer"
                  onClick={() => onSelectRun(execution)}
                  data-state={
                    selectedRun?.id == execution.id ? "selected" : null
                  }
                >
                  <TableCell>
                    <TextRenderer
                      value={
                        flows.find((f) => f.graph_id == execution.graph_id)
                          ?.name
                      }
                      truncateLengthLimit={30}
                    />
                  </TableCell>
                  <TableCell>
                    {moment(execution.started_at).format("HH:mm")}
                  </TableCell>
                  <TableCell>
                    <FlowRunStatusBadge
                      status={execution.status}
                      className="w-full justify-center"
                    />
                  </TableCell>
                  <TableCell>
                    {execution.stats
                      ? formatDuration(execution.stats.duration)
                      : ""}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

function formatDuration(seconds: number): string {
  return (
    (seconds < 100 ? seconds.toPrecision(2) : Math.round(seconds)).toString() +
    "s"
  );
}

export default FlowRunsList;

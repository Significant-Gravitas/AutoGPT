import { GraphExecutionMeta, LibraryAgent } from "@/lib/autogpt-server-api";
import React from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/__legacy__/ui/card";
import { Button } from "@/components/__legacy__/ui/button";
import { TextRenderer } from "@/components/__legacy__/ui/render";
import Link from "next/link";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTrigger,
} from "@/components/__legacy__/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/__legacy__/ui/dropdown-menu";
// Phosphor Icons — project standard (replaced @radix-ui/react-icons)
import { CaretDown, ArrowSquareIn, Robot } from "@phosphor-icons/react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/__legacy__/ui/table";
import moment from "moment/moment";
import { DialogTitle } from "@/components/__legacy__/ui/dialog";
import { AgentImportForm } from "./AgentImportForm";

interface Props {
  flows: LibraryAgent[];
  executions?: GraphExecutionMeta[];
  selectedFlow: LibraryAgent | null;
  onSelectFlow: (f: LibraryAgent) => void;
  className?: string;
  isLoading?: boolean;
  error?: Error | null;
}

export function AgentFlowList({
  flows,
  executions,
  selectedFlow,
  onSelectFlow,
  className,
  isLoading,
  error,
}: Props) {
  return (
    <Card className={className}>
      <CardHeader className="flex-row items-center justify-between space-x-3 space-y-0">
        <CardTitle>Agents</CardTitle>

        <div className="flex items-center">
          {/* Split "Create" button */}
          <Button variant="outline" className="rounded-r-none">
            <Link href="/build">Create</Link>
          </Button>
          <Dialog>
            {/* https://ui.shadcn.com/docs/components/dialog#notes */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  className={"rounded-l-none border-l-0 px-2"}
                  data-testid="create-agent-dropdown"
                >
                  <CaretDown size={14} />
                </Button>
              </DropdownMenuTrigger>

              <DropdownMenuContent>
                <DialogTrigger asChild>
                  <DropdownMenuItem data-testid="import-agent-from-file">
                    <ArrowSquareIn size={16} className="mr-2" /> Import from
                    file
                  </DropdownMenuItem>
                </DialogTrigger>
              </DropdownMenuContent>
            </DropdownMenu>

            <DialogContent>
              <DialogHeader>
                <DialogTitle className="sr-only">Import Agent</DialogTitle>
                <h2 className="text-lg font-semibold">
                  Import an Agent from a file
                </h2>
              </DialogHeader>
              <AgentImportForm />
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>

      <CardContent>
        {error && (
          <p className="py-4 text-center text-sm text-destructive">
            Failed to load agents. Please try again.
          </p>
        )}

        {!error && !isLoading && flows.length === 0 && (
          <div className="flex flex-col items-center gap-3 py-10 text-center text-muted-foreground">
            <Robot size={40} weight="thin" />
            <p className="text-sm font-medium">No agents yet</p>
            <p className="max-w-[200px] text-xs">
              Create your first agent in the workflow builder.
            </p>
            <Button asChild variant="outline" size="sm" className="mt-1">
              <Link href="/build">Open Builder</Link>
            </Button>
          </div>
        )}

        {(isLoading || flows.length > 0) && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                {executions && (
                  <TableHead className="md:hidden lg:table-cell">
                    # of runs
                  </TableHead>
                )}
                {executions && <TableHead>Last run</TableHead>}
              </TableRow>
            </TableHeader>
            <TableBody data-testid="agent-flow-list-body">
              {flows
                .map((flow) => {
                  let runCount = 0,
                    lastRun: GraphExecutionMeta | null = null;
                  if (executions) {
                    const _flowRuns = executions.filter(
                      (r) => r.graph_id == flow.graph_id,
                    );
                    runCount = _flowRuns.length;
                    lastRun =
                      runCount == 0
                        ? null
                        : _flowRuns.reduce((a, c) => {
                            const aTime = a.started_at?.getTime() ?? 0;
                            const cTime = c.started_at?.getTime() ?? 0;
                            return aTime > cTime ? a : c;
                          });
                  }
                  return { flow, runCount, lastRun };
                })
                .sort((a, b) => {
                  if (!a.lastRun && !b.lastRun) return 0;
                  if (!a.lastRun) return 1;
                  if (!b.lastRun) return -1;
                  const bTime = b.lastRun.started_at?.getTime() ?? 0;
                  const aTime = a.lastRun.started_at?.getTime() ?? 0;
                  return bTime - aTime;
                })
                .map(({ flow, runCount, lastRun }) => (
                  <TableRow
                    key={flow.id}
                    data-testid={flow.id}
                    data-name={flow.name}
                    className="cursor-pointer"
                    onClick={() => onSelectFlow(flow)}
                    data-state={selectedFlow?.id == flow.id ? "selected" : null}
                  >
                    <TableCell>
                      <TextRenderer
                        value={flow.name}
                        truncateLengthLimit={30}
                      />
                    </TableCell>
                    {executions && (
                      <TableCell className="md:hidden lg:table-cell">
                        {runCount}
                      </TableCell>
                    )}
                    {executions &&
                      (!lastRun ? (
                        <TableCell />
                      ) : (
                        <TableCell
                          title={moment(lastRun.started_at).toString()}
                        >
                          {moment(lastRun.started_at).fromNow()}
                        </TableCell>
                      ))}
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
export default AgentFlowList;

import AutoGPTServerAPI, { GraphMeta } from "@/lib/autogpt-server-api";
import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDownIcon, EnterIcon } from "@radix-ui/react-icons";
import { AgentImportForm } from "@/components/agent-import-form";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import moment from "moment/moment";
import { FlowRun } from "@/lib/types";
import { DialogTitle } from "@/components/ui/dialog";

export const AgentFlowList = ({
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
  const api = useMemo(() => new AutoGPTServerAPI(), []);
  useEffect(() => {
    api.listTemplates().then((templates) => setTemplates(templates));
  }, [api]);

  return (
    <Card className={className}>
      <CardHeader className="flex-row items-center justify-between space-x-3 space-y-0">
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
              <DialogHeader>
                <DialogTitle className="sr-only">Import Agent</DialogTitle>
                <h2 className="text-lg font-semibold">
                  Import an Agent (template) from a file
                </h2>
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
export default AgentFlowList;

import React, { useEffect, useMemo, useState } from "react";
import AutoGPTServerAPI, {
  GraphExecution,
  Graph,
  GraphMeta,
  safeCopyGraph,
} from "@/lib/autogpt-server-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button, buttonVariants } from "@/components/ui/button";
import { ClockIcon, ExitIcon, Pencil2Icon } from "@radix-ui/react-icons";
import Link from "next/link";
import { exportAsJSONFile } from "@/lib/utils";
import { FlowRunsStats } from "@/components/monitor/index";
import { Trash2Icon } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { useIndexedDB } from "@/hooks/useIndexedDB";

export const FlowInfo: React.FC<
  React.HTMLAttributes<HTMLDivElement> & {
    flow: GraphMeta;
    executions: GraphExecution[];
    flowVersion?: number | "all";
    refresh: () => void;
  }
> = ({ flow, executions, flowVersion, refresh, ...props }) => {
  const api = useMemo(() => new AutoGPTServerAPI(), []);
  const { clearStore } = useIndexedDB(flow.id);

  const [flowVersions, setFlowVersions] = useState<Graph[] | null>(null);
  const [selectedVersion, setSelectedFlowVersion] = useState(
    flowVersion ?? "all",
  );
  const selectedFlowVersion: Graph | undefined = flowVersions?.find(
    (v) =>
      v.version == (selectedVersion == "all" ? flow.version : selectedVersion),
  );

  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  useEffect(() => {
    api.getGraphAllVersions(flow.id).then((result) => setFlowVersions(result));
  }, [flow.id, api]);

  return (
    <Card {...props}>
      <CardHeader className="flex-row justify-between space-x-3 space-y-0">
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
            className={buttonVariants({ variant: "default" })}
            href={`/build?flowID=${flow.id}`}
          >
            <Pencil2Icon className="mr-2" />
            Open in Builder
          </Link>
          <Button
            variant="outline"
            className="px-2.5"
            title="Export to a JSON-file"
            data-testid="export-button"
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
            <ExitIcon className="mr-2" /> Export
          </Button>
          <Button
            variant="outline"
            onClick={() => setIsDeleteModalOpen(true)}
            data-testid="delete-button"
          >
            <Trash2Icon className="h-full" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <FlowRunsStats
          flows={[selectedFlowVersion ?? flow]}
          executions={executions.filter(
            (execution) =>
              execution.graph_id == flow.id &&
              (selectedVersion == "all" ||
                execution.graph_version == selectedVersion),
          )}
        />
      </CardContent>
      <Dialog open={isDeleteModalOpen} onOpenChange={setIsDeleteModalOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Agent</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this agent? <br />
              This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsDeleteModalOpen(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                api.deleteGraph(flow.id).then(async () => {
                  setIsDeleteModalOpen(false);
                  await clearStore();
                  refresh();
                });
              }}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
};
export default FlowInfo;

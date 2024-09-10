import React, { useEffect, useState } from "react";
import AutoGPTServerAPI, {
  Graph,
  GraphMeta,
  safeCopyGraph,
} from "@/lib/autogpt-server-api";
import { FlowRun } from "@/lib/types";
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

export const FlowInfo: React.FC<
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
export default FlowInfo;

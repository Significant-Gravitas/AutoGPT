import React from "react";
import moment from "moment";
import { MoreVertical } from "lucide-react";

import { cn } from "@/lib/utils";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import AgentRunStatusChip, {
  AgentRunStatus,
} from "@/components/agents/agent-run-status-chip";

export type AgentRunSummaryProps = {
  status: AgentRunStatus;
  title: string;
  timestamp: number | Date;
  selected?: boolean;
  onClick?: () => void;
  // onRename: () => void;
  onDelete: () => void;
  className?: string;
};

export default function AgentRunSummaryCard({
  status,
  title,
  timestamp,
  selected = false,
  onClick,
  // onRename,
  onDelete,
  className,
}: AgentRunSummaryProps): React.ReactElement {
  return (
    <Card
      className={cn(
        "agpt-rounded-card cursor-pointer border-zinc-300",
        selected ? "agpt-card-selected" : "",
        className,
      )}
      onClick={onClick}
    >
      <CardContent className="relative p-2.5 lg:p-4">
        <AgentRunStatusChip status={status} />

        <div className="mt-5 flex items-center justify-between">
          <h3 className="truncate pr-2 text-base font-medium text-neutral-900">
            {title}
          </h3>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="h-5 w-5 p-0">
                <MoreVertical className="h-5 w-5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              {/* {onPinAsPreset && (
                <DropdownMenuItem onClick={onPinAsPreset}>
                  Pin as a preset
              </DropdownMenuItem>
              )} */}

              {/* <DropdownMenuItem onClick={onRename}>Rename</DropdownMenuItem> */}

              <DropdownMenuItem onClick={onDelete}>Delete</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        <p
          className="mt-1 text-sm font-normal text-neutral-500"
          title={moment(timestamp).toString()}
        >
          Ran {moment(timestamp).fromNow()}
        </p>
      </CardContent>
    </Card>
  );
}

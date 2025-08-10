import React from "react";
import moment from "moment";

import { cn } from "@/lib/utils";

import { Link2Icon, Link2OffIcon, MoreVertical } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import AgentRunStatusChip, {
  AgentRunStatus,
} from "@/components/agents/agent-run-status-chip";
import AgentStatusChip, {
  AgentStatus,
} from "@/components/agents/agent-status-chip";

export type AgentRunSummaryProps = (
  | {
      type: "run";
      status: AgentRunStatus;
    }
  | {
      type: "preset";
      status: AgentStatus;
    }
  | {
      type: "schedule";
      status: "scheduled";
    }
) & {
  title: string;
  timestamp?: number | Date;
  selected?: boolean;
  onClick?: () => void;
  // onRename: () => void;
  onDelete: () => void;
  className?: string;
};

export default function AgentRunSummaryCard({
  type,
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
        {(type == "run" || type == "schedule") && (
          <AgentRunStatusChip status={status} />
        )}
        {type == "preset" && (
          <div className="flex items-center justify-between">
            <AgentStatusChip status={status} />

            <div className="flex items-center text-sm text-zinc-400">
              {status == "inactive" ? (
                <Link2OffIcon className="mr-1 size-4" />
              ) : (
                <Link2Icon className="mr-1 size-4" />
              )}{" "}
              Trigger
            </div>
          </div>
        )}

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

        {timestamp && (
          <p
            className="mt-1 text-sm font-normal text-neutral-500"
            title={moment(timestamp).toString()}
          >
            {moment(timestamp).isBefore() ? "Ran" : "Runs in"}{" "}
            {moment(timestamp).fromNow()}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

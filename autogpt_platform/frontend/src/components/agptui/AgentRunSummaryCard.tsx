import React from "react";
import moment from "moment";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { MoreVertical } from "lucide-react";
import { AgentRunStatusChip, AgentRunStatus } from "./AgentRunStatusChip";

export type AgentRunSummaryProps = {
  agentID: string;
  agentRunID: string;
  status: AgentRunStatus;
  title: string;
  timestamp: Date;
};

export function AgentRunSummaryCard({
  agentID,
  agentRunID,
  status,
  title,
  timestamp,
}: AgentRunSummaryProps): React.ReactElement {
  return (
    <Card className="rounded-lg border-zinc-300">
      <CardContent className="relative p-4">
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
              <DropdownMenuItem /* TODO: implement */>
                Pin into a template
              </DropdownMenuItem>
              <DropdownMenuItem /* TODO: implement */>Rename</DropdownMenuItem>
              <DropdownMenuItem /* TODO: implement */>Delete</DropdownMenuItem>
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

export default AgentRunSummaryCard;

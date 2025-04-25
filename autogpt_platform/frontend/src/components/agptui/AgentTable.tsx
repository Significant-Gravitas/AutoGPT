"use client";

import * as React from "react";
import { AgentTableRow, AgentTableRowProps } from "./AgentTableRow";
import { AgentTableCard } from "./AgentTableCard";
import { StoreSubmissionRequest } from "@/lib/autogpt-server-api/types";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";

export interface AgentTableProps {
  agents: Omit<
    AgentTableRowProps,
    | "setSelectedAgents"
    | "selectedAgents"
    | "onEditSubmission"
    | "onDeleteSubmission"
  >[];
  onEditSubmission: (submission: StoreSubmissionRequest) => void;
  onDeleteSubmission: (submission_id: string) => void;
}

export const AgentTable: React.FC<AgentTableProps> = ({
  agents,
  onEditSubmission,
  onDeleteSubmission,
}) => {
  // Use state to track selected agents
  const [selectedAgents, setSelectedAgents] = React.useState<Set<string>>(
    new Set(),
  );

  // Handle select all checkbox
  const handleSelectAll = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.checked) {
        setSelectedAgents(new Set(agents.map((agent) => agent.agent_id)));
      } else {
        setSelectedAgents(new Set());
      }
    },
    [agents],
  );

  return (
    <div className="w-full">
      {/* Table for desktop view */}
      <div className="hidden md:block">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>
                <Checkbox
                  id="selectAllAgents"
                  aria-label="Select all agents"
                  checked={
                    selectedAgents.size === agents.length && agents.length > 0
                  }
                  onCheckedChange={(checked) => {
                    if (checked) {
                      setSelectedAgents(
                        new Set(agents.map((agent) => agent.agent_id)),
                      );
                    } else {
                      setSelectedAgents(new Set());
                    }
                  }}
                />
              </TableHead>
              <TableHead className="font-sans text-sm font-medium text-neutral-800">
                Agent info
              </TableHead>
              <TableHead className="font-sans text-sm font-medium text-neutral-800">
                Date submitted
              </TableHead>
              <TableHead className="font-sans text-sm font-medium text-neutral-800">
                Status
              </TableHead>
              <TableHead className="font-sans text-sm font-medium text-neutral-800">
                Runs
              </TableHead>
              <TableHead className="text-right font-sans text-sm font-medium text-neutral-800">
                Reviews
              </TableHead>
              <TableHead className="font-sans text-sm font-medium text-neutral-800"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {agents.length > 0 ? (
              agents.map((agent) => (
                <AgentTableRow
                  key={agent.id}
                  {...agent}
                  selectedAgents={selectedAgents}
                  setSelectedAgents={setSelectedAgents}
                  onEditSubmission={onEditSubmission}
                  onDeleteSubmission={onDeleteSubmission}
                />
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={7} className="py-4 text-center">
                  <span className="font-sans text-base text-neutral-600 dark:text-neutral-400">
                    No agents available. Create your first agent to get started!
                  </span>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      {/* Mobile view with cards */}
      <div className="md:hidden">
        {agents.length > 0 ? (
          <div className="flex flex-col">
            {agents.map((agent) => (
              <AgentTableCard
                key={agent.id}
                {...agent}
                onEditSubmission={onEditSubmission}
              />
            ))}
          </div>
        ) : (
          <div className="py-4 text-center font-sans text-base text-neutral-600 dark:text-neutral-400">
            No agents available. Create your first agent to get started!
          </div>
        )}
      </div>
    </div>
  );
};

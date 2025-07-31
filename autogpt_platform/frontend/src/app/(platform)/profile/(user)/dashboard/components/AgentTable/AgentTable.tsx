"use client";

import * as React from "react";
import { AgentTableCard } from "../AgentTableCard/AgentTableCard";
import { StoreSubmissionRequest } from "@/app/api/__generated__/models/storeSubmissionRequest";
import {
  AgentTableRow,
  AgentTableRowProps,
} from "../AgentTableRow/AgentTableRow";

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
  return (
    <div className="w-full">
      {/* Table header - Hide on mobile */}
      <div className="hidden flex-col md:flex">
        <div className="border-t border-neutral-300 dark:border-neutral-700" />
        <div className="flex items-center px-4 py-2">
          <div className="grid w-full grid-cols-[minmax(400px,1fr),180px,140px,100px,100px,40px] items-center gap-4">
            <div className="text-sm font-medium text-neutral-800 dark:text-neutral-200">
              Agent info
            </div>
            <div className="text-sm font-medium text-neutral-800 dark:text-neutral-200">
              Date submitted
            </div>
            <div className="text-sm font-medium text-neutral-800 dark:text-neutral-200">
              Status
            </div>
            <div className="text-right text-sm font-medium text-neutral-800 dark:text-neutral-200">
              Runs
            </div>
            <div className="text-right text-sm font-medium text-neutral-800 dark:text-neutral-200">
              Reviews
            </div>
            <div></div>
          </div>
        </div>
        <div className="border-b border-neutral-300 dark:border-neutral-700" />
      </div>

      {/* Table body */}
      {agents.length > 0 ? (
        <div className="flex flex-col">
          {agents.map((agent) => (
            <div key={agent.id} className="md:block">
              <AgentTableRow
                {...agent}
                onEditSubmission={onEditSubmission}
                onDeleteSubmission={onDeleteSubmission}
              />
              <div className="block md:hidden">
                <AgentTableCard
                  {...agent}
                  onEditSubmission={onEditSubmission}
                />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="py-4 text-center font-sans text-base text-neutral-600 dark:text-neutral-400">
          No agents available. Create your first agent to get started!
        </div>
      )}
    </div>
  );
};

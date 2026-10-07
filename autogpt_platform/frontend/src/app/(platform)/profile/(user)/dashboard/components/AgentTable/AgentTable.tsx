"use client";

import * as React from "react";
import { AgentTableCard } from "../AgentTableCard/AgentTableCard";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { AgentTableRow } from "../AgentTableRow/AgentTableRow";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import { Text } from "@/components/atoms/Text/Text";

export interface AgentTableProps {
  storeAgentSubmissions: StoreSubmission[];
  onViewSubmission: (submission: StoreSubmission) => void;
  onDeleteSubmission: (submission_id: string) => void;
  onEditSubmission: (
    submission: StoreSubmissionEditRequest & {
      store_listing_version_id: string | undefined;
      graph_id: string;
    },
  ) => void;
}

export const AgentTable: React.FC<AgentTableProps> = ({
  storeAgentSubmissions,
  onViewSubmission,
  onDeleteSubmission,
  onEditSubmission,
}) => {
  return (
    <div className="w-full" data-testid="agent-table">
      {/* Table header - Hide on mobile */}
      <div className="hidden flex-col md:flex">
        <div className="border-t border-zinc-300" />
        <div className="flex items-center px-4 py-2">
          <div className="grid w-full grid-cols-[minmax(400px,1fr)_180px_140px_100px_100px_40px] items-center gap-4">
            <Text variant="body-medium" as="div" tone="primary">
              Agent info
            </Text>
            <Text variant="body-medium" as="div" tone="primary">
              Date submitted
            </Text>
            <Text variant="body-medium" as="div" tone="primary">
              Status
            </Text>
            <Text
              variant="body-medium"
              as="div"
              tone="primary"
              className="text-right"
            >
              Runs
            </Text>
            <Text
              variant="body-medium"
              as="div"
              tone="primary"
              className="text-right"
            >
              Reviews
            </Text>
            <div></div>
          </div>
        </div>
        <div className="border-b border-zinc-300" />
      </div>

      {/* Table body */}
      {storeAgentSubmissions.length > 0 ? (
        <div className="flex flex-col">
          {storeAgentSubmissions.map((agentSubmission) => (
            <div key={agentSubmission.listing_version_id} className="md:block">
              <AgentTableRow
                storeAgentSubmission={agentSubmission}
                onViewSubmission={onViewSubmission}
                onDeleteSubmission={onDeleteSubmission}
                onEditSubmission={onEditSubmission}
              />
              <div className="block md:hidden">
                <AgentTableCard
                  storeAgentSubmission={agentSubmission}
                  onViewSubmission={onViewSubmission}
                />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <Text
          variant="large"
          as="div"
          tone="secondary"
          className="py-4 text-center"
        >
          No agents available. Create your first agent to get started!
        </Text>
      )}
    </div>
  );
};

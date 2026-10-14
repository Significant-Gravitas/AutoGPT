"use client";

import { ExpertAvatar } from "@/app/(platform)/copilot/components/ChatMessagesContainer/components/ExpertAvatar/ExpertAvatar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

export interface AssigneeIdentity {
  name: string;
  roleLabel: string;
  avatarUrl: string | null;
  color?: string | null;
  isAutopilot: boolean;
}

interface Props {
  assignee: AssigneeIdentity;
  onReassign?: () => void;
}

export function AssigneeButton({ assignee, onReassign }: Props) {
  const avatar = (
    <ExpertAvatar
      name={assignee.name}
      avatarUrl={assignee.avatarUrl}
      color={assignee.color}
      isAutopilot={assignee.isAutopilot}
      size="sm"
    />
  );

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {onReassign ? (
          <button
            type="button"
            onClick={onReassign}
            aria-label={`Reassign from ${assignee.name}`}
            className="flex size-8 shrink-0 items-center justify-center rounded-full transition-colors hover:bg-zinc-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800"
          >
            {avatar}
          </button>
        ) : (
          <span
            tabIndex={0}
            aria-label={`Assigned to ${assignee.name}`}
            className="flex size-8 shrink-0 items-center justify-center rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800"
          >
            {avatar}
          </span>
        )}
      </TooltipTrigger>
      <TooltipContent side="bottom">
        {assignee.name} · {assignee.roleLabel}
        {onReassign && " — click to reassign"}
      </TooltipContent>
    </Tooltip>
  );
}

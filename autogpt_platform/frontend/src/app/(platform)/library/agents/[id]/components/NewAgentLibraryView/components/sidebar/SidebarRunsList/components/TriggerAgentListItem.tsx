"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { RobotIcon } from "@phosphor-icons/react";
import { formatDistanceToNow } from "date-fns";
import { IconWrapper } from "./IconWrapper";
import { SidebarItemCard } from "./SidebarItemCard";
import { TriggerAgentActionsDropdown } from "./TriggerAgentActionsDropdown";

interface Props {
  triggerAgent: LibraryAgent;
  parentAgent: LibraryAgent;
  selected?: boolean;
  onClick?: () => void;
  onDeleted?: () => void;
}

export function TriggerAgentListItem({
  triggerAgent,
  parentAgent,
  selected,
  onClick,
  onDeleted,
}: Props) {
  return (
    <SidebarItemCard
      icon={
        <IconWrapper className="border-blue-50 bg-blue-50">
          <RobotIcon size={16} className="text-zinc-700" weight="bold" />
        </IconWrapper>
      }
      title={triggerAgent.name}
      description={`Updated ${formatDistanceToNow(triggerAgent.updated_at, {
        addSuffix: true,
      })}`}
      onClick={onClick}
      selected={selected}
      actions={
        <TriggerAgentActionsDropdown
          parentAgent={parentAgent}
          triggerAgent={triggerAgent}
          onDeleted={onDeleted}
        />
      }
    />
  );
}

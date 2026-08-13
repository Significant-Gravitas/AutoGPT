"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { formatDistanceToNow } from "date-fns";
import { IconWrapper } from "./IconWrapper";
import { SidebarItemCard } from "./SidebarItemCard";
import { TriggerAgentActionsDropdown } from "./TriggerAgentActionsDropdown";
import { Robot01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
          <Icon icon={Robot01Icon} size={16} className="text-zinc-700" />
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

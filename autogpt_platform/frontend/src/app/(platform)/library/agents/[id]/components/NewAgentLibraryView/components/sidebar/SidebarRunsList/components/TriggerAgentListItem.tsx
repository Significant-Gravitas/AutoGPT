"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { IconWrapper } from "./IconWrapper";
import { SidebarItemCard } from "./SidebarItemCard";
import { TriggerAgentActionsDropdown } from "./TriggerAgentActionsDropdown";
import { Robot01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";

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
        <IconWrapper className="border-info-surface bg-info-surface text-foreground-body">
          <Icon icon={Robot01Icon} size={16} />
        </IconWrapper>
      }
      title={triggerAgent.name}
      description={
        <>
          Updated <RelativeTime date={triggerAgent.updated_at} />
        </>
      }
      onClick={onClick}
      selected={selected}
      testID={`library-trigger-agent-item-${triggerAgent.id}`}
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

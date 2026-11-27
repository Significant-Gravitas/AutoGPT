"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { IconWrapper } from "./IconWrapper";
import { SidebarItemCard } from "./SidebarItemCard";
import { TriggerActionsDropdown } from "./TriggerActionsDropdown";
import { FlashIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";

interface Props {
  trigger: LibraryAgentPreset;
  agent: LibraryAgent;
  selected?: boolean;
  onClick?: () => void;
  onDeleted?: () => void;
}

export function TriggerListItem({
  trigger,
  agent,
  selected,
  onClick,
  onDeleted,
}: Props) {
  return (
    <SidebarItemCard
      icon={
        <IconWrapper className="border-accent-surface bg-accent-surface text-foreground-body">
          <Icon icon={FlashIcon} size={16} />
        </IconWrapper>
      }
      title={trigger.name}
      description={<RelativeTime date={trigger.updated_at} />}
      onClick={onClick}
      selected={selected}
      testID={`library-trigger-item-${trigger.id}`}
      actions={
        <TriggerActionsDropdown
          agent={agent}
          trigger={trigger}
          onDeleted={onDeleted}
        />
      }
    />
  );
}

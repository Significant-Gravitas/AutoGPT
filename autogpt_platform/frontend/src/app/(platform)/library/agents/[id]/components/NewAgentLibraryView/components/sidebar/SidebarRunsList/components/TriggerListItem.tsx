"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { formatDistanceToNow } from "date-fns";
import { IconWrapper } from "./IconWrapper";
import { SidebarItemCard } from "./SidebarItemCard";
import { TriggerActionsDropdown } from "./TriggerActionsDropdown";
import { FlashIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
        <IconWrapper className="border-purple-50 bg-purple-50">
          <Icon icon={FlashIcon} size={16} className="text-zinc-700" />
        </IconWrapper>
      }
      title={trigger.name}
      description={formatDistanceToNow(trigger.updated_at, { addSuffix: true })}
      onClick={onClick}
      selected={selected}
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

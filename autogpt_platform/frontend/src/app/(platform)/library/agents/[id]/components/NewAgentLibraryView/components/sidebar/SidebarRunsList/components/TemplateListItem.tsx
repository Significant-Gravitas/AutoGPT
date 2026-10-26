"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { formatDistanceToNow } from "date-fns";
import { IconWrapper } from "./IconWrapper";
import { SidebarItemCard } from "./SidebarItemCard";
import { TemplateActionsDropdown } from "./TemplateActionsDropdown";
import { File02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  template: LibraryAgentPreset;
  agent: LibraryAgent;
  selected?: boolean;
  onClick?: () => void;
  onDeleted?: () => void;
}

export function TemplateListItem({
  template,
  agent,
  selected,
  onClick,
  onDeleted,
}: Props) {
  return (
    <SidebarItemCard
      icon={
        <IconWrapper className="border-info-surface bg-info-surface">
          <Icon icon={File02Icon} size={16} className="text-foreground-body" />
        </IconWrapper>
      }
      title={template.name}
      description={formatDistanceToNow(template.updated_at, {
        addSuffix: true,
      })}
      onClick={onClick}
      selected={selected}
      actions={
        <TemplateActionsDropdown
          agent={agent}
          template={template}
          onDeleted={onDeleted}
        />
      }
    />
  );
}

"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { IconWrapper } from "./IconWrapper";
import { SidebarItemCard } from "./SidebarItemCard";
import { TemplateActionsDropdown } from "./TemplateActionsDropdown";
import { File02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";

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
        <IconWrapper className="border-info-surface bg-info-surface text-foreground-body">
          <Icon icon={File02Icon} size={16} />
        </IconWrapper>
      }
      title={template.name}
      description={<RelativeTime date={template.updated_at} />}
      onClick={onClick}
      selected={selected}
      testID={`library-template-item-${template.id}`}
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

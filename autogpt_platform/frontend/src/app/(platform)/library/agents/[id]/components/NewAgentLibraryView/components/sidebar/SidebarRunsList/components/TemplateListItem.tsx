"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { FileTextIcon } from "@phosphor-icons/react";
import moment from "moment";
import { IconWrapper } from "./IconWrapper";
import { SidebarItemCard } from "./SidebarItemCard";
import { TemplateActionsDropdown } from "./TemplateActionsDropdown";

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
        <IconWrapper className="border-blue-50 bg-blue-50">
          <FileTextIcon size={16} className="text-zinc-700" weight="bold" />
        </IconWrapper>
      }
      title={template.name}
      description={moment(template.updated_at).fromNow()}
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

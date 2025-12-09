"use client";

import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { FileTextIcon } from "@phosphor-icons/react";
import moment from "moment";
import { IconWrapper } from "./RunIconWrapper";
import { RunSidebarCard } from "./RunSidebarCard";

interface TemplateListItemProps {
  template: LibraryAgentPreset;
  selected?: boolean;
  onClick?: () => void;
}

export function TemplateListItem({
  template,
  selected,
  onClick,
}: TemplateListItemProps) {
  return (
    <RunSidebarCard
      icon={
        <IconWrapper className="border-blue-50 bg-blue-50">
          <FileTextIcon size={16} className="text-zinc-700" weight="bold" />
        </IconWrapper>
      }
      title={template.name}
      description={moment(template.updated_at).fromNow()}
      onClick={onClick}
      selected={selected}
    />
  );
}

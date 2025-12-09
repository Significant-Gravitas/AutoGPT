"use client";

import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { LightningIcon } from "@phosphor-icons/react";
import moment from "moment";
import { IconWrapper } from "./RunIconWrapper";
import { RunSidebarCard } from "./RunSidebarCard";

interface TriggerListItemProps {
  trigger: LibraryAgentPreset;
  selected?: boolean;
  onClick?: () => void;
}

export function TriggerListItem({
  trigger,
  selected,
  onClick,
}: TriggerListItemProps) {
  return (
    <RunSidebarCard
      icon={
        <IconWrapper className="border-purple-50 bg-purple-50">
          <LightningIcon size={16} className="text-zinc-700" weight="bold" />
        </IconWrapper>
      }
      title={trigger.name}
      description={moment(trigger.updated_at).fromNow()}
      onClick={onClick}
      selected={selected}
    />
  );
}

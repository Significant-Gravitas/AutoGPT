"use client";

import { getExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { SidebarMenu } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { ReactNode, useState } from "react";

export const EXPERT_GROUP_PREVIEW_COUNT = 6;

interface Props {
  label: string;
  avatarUrl: string | null;
  role: string | null;
  sessions: SessionSummaryResponse[];
  renderItem: (session: SessionSummaryResponse) => ReactNode;
}

export function ExpertChatGroup({
  label,
  avatarUrl,
  role,
  sessions,
  renderItem,
}: Props) {
  const [visibleCount, setVisibleCount] = useState(EXPERT_GROUP_PREVIEW_COUNT);
  const visibleSessions = sessions.slice(0, visibleCount);
  const hasHiddenSessions = sessions.length > visibleSessions.length;

  return (
    <Collapsible defaultOpen className="group/expert-group">
      <CollapsibleTrigger
        aria-label={`${label} chats`}
        className="mb-1 flex w-full items-center gap-2 rounded-md px-2 py-0.5 text-left text-[13px] font-medium text-zinc-900 hover:bg-zinc-100"
      >
        <Avatar className="h-5 w-5">
          {avatarUrl ? <AvatarImage src={avatarUrl} alt={label} /> : null}
          <AvatarFallback className="text-[9px]">{label}</AvatarFallback>
        </Avatar>
        <span className="truncate">{label}</span>
        {role ? (
          <span
            className={cn(
              "shrink-0 rounded-full px-1.5 py-px text-[10px] font-medium",
              getExpertAccent(role).pill,
            )}
          >
            {role}
          </span>
        ) : null}
        <Icon
          icon={ArrowDown01Icon}
          className="ease-[cubic-bezier(0.33,1,0.68,1)] ml-auto size-4 shrink-0 text-zinc-400 transition-transform duration-200 group-data-[state=open]/expert-group:rotate-180 motion-reduce:transition-none"
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down motion-reduce:animate-none">
        <div className="ml-[17px] border-l border-zinc-200 pl-1.5">
          <SidebarMenu>{visibleSessions.map(renderItem)}</SidebarMenu>
          {hasHiddenSessions && (
            <button
              type="button"
              onClick={() =>
                setVisibleCount((count) => count + EXPERT_GROUP_PREVIEW_COUNT)
              }
              className="mt-0.5 w-full rounded-md px-2 py-1 text-left text-xs font-medium text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800"
            >
              Load more
            </button>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

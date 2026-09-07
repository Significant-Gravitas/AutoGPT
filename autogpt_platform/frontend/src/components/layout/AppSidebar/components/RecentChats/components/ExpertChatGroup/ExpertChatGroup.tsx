"use client";

import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { SidebarMenu } from "@/components/ui/sidebar";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { ReactNode, useState } from "react";

export const EXPERT_GROUP_PREVIEW_COUNT = 10;

interface Props {
  label: string;
  avatarUrl: string | null;
  sessions: SessionSummaryResponse[];
  renderItem: (session: SessionSummaryResponse) => ReactNode;
}

export function ExpertChatGroup({
  label,
  avatarUrl,
  sessions,
  renderItem,
}: Props) {
  const [isOpen, setIsOpen] = useState(true);
  const [visibleCount, setVisibleCount] = useState(EXPERT_GROUP_PREVIEW_COUNT);
  const visibleSessions = sessions.slice(0, visibleCount);
  const hasHiddenSessions = sessions.length > visibleSessions.length;
  const runningSessions = sessions.filter((session) => session.is_processing);

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className="group/expert-group"
    >
      <CollapsibleTrigger
        aria-label={`${label} chats`}
        className="mb-1 flex w-full items-center gap-2 rounded-md px-2 py-0.5 text-left text-sm font-medium text-zinc-900 hover:bg-zinc-100"
      >
        <Avatar className="h-6 w-6">
          {avatarUrl ? (
            <AvatarImage src={avatarUrl} alt={label} width={48} height={48} />
          ) : null}
          <AvatarFallback>
            <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-4" />
          </AvatarFallback>
        </Avatar>
        <span className="truncate">{label}</span>
        <Icon
          icon={ArrowDown01Icon}
          className="ease-[cubic-bezier(0.33,1,0.68,1)] ml-auto size-5 shrink-0 text-zinc-400 transition-transform duration-200 group-data-[state=open]/expert-group:rotate-180 motion-reduce:transition-none"
        />
      </CollapsibleTrigger>

      {!isOpen && runningSessions.length > 0 && (
        <GroupBody>
          <SidebarMenu>{runningSessions.map(renderItem)}</SidebarMenu>
        </GroupBody>
      )}

      <CollapsibleContent className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down motion-reduce:animate-none">
        <GroupBody>
          <SidebarMenu>{visibleSessions.map(renderItem)}</SidebarMenu>
          {hasHiddenSessions && (
            <button
              type="button"
              aria-label={`Load more ${label} chats`}
              onClick={() =>
                setVisibleCount((count) => count + EXPERT_GROUP_PREVIEW_COUNT)
              }
              className="mt-0.5 w-full rounded-md px-2 py-1 text-left text-xs font-medium text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800"
            >
              Load more
            </button>
          )}
        </GroupBody>
      </CollapsibleContent>
    </Collapsible>
  );
}

function GroupBody({ children }: { children: ReactNode }) {
  return (
    <div className="relative ml-[17px] pl-1.5 before:absolute before:inset-y-0 before:left-0 before:w-px before:bg-gradient-to-b before:from-zinc-200/70 before:to-transparent">
      {children}
    </div>
  );
}

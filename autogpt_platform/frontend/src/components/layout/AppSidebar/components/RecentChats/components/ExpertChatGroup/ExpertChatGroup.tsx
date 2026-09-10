"use client";

import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Icon } from "@/components/atoms/Icon/Icon";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { SidebarMenu } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import Link, { useLinkStatus } from "next/link";
import { ReactNode, useState } from "react";

export const EXPERT_GROUP_PREVIEW_COUNT = 10;

// The chevron, the slot the header reserves next to it, and the new-chat link
// that floats over that slot are all one size.
const CHEVRON_SIZE_CLASS = "size-5";
// Where that slot starts, measured in from the header's right edge: the
// trigger's `px-2` (0.5rem) + the chevron's `size-5` (1.25rem) + the `gap-2`
// between them (0.5rem) = 2.25rem. Change any of the three and this moves.
const NEW_CHAT_LINK_OFFSET_CLASS = "right-9";

interface Props {
  label: string;
  avatarUrl: string | null;
  newChatHref: string | null;
  sessions: SessionSummaryResponse[];
  renderItem: (session: SessionSummaryResponse) => ReactNode;
}

export function ExpertChatGroup({
  label,
  avatarUrl,
  newChatHref,
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
      {/* The trigger is itself a button, so the new-chat link sits beside it
          and floats over a slot the trigger reserves before the chevron —
          long labels then truncate against the link rather than under it. */}
      <div className="group/expert-header relative mb-1 flex items-center rounded-md hover:bg-zinc-100">
        <CollapsibleTrigger
          aria-label={`${label} chats`}
          className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-0.5 text-left text-sm font-medium text-zinc-900"
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
          {newChatHref && (
            <span
              aria-hidden
              className={cn("ml-auto shrink-0", CHEVRON_SIZE_CLASS)}
            />
          )}
          <Icon
            icon={ArrowDown01Icon}
            className={cn(
              "ease-[cubic-bezier(0.33,1,0.68,1)] shrink-0 text-zinc-400 transition-transform duration-200 group-data-[state=open]/expert-group:rotate-180 motion-reduce:transition-none",
              CHEVRON_SIZE_CLASS,
              !newChatHref && "ml-auto",
            )}
          />
        </CollapsibleTrigger>
        {newChatHref && <NewChatLink href={newChatHref} label={label} />}
      </div>

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

function NewChatLink({ href, label }: { href: string; label: string }) {
  const name = `New chat with ${label}`;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Link
          href={href}
          aria-label={name}
          className={cn(
            "absolute top-1/2 flex -translate-y-1/2 items-center justify-center rounded-md text-zinc-500 transition-opacity group-focus-within/expert-header:opacity-100 group-hover/expert-header:opacity-100 hover:bg-zinc-200 hover:text-zinc-900 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 md:opacity-0",
            CHEVRON_SIZE_CLASS,
            NEW_CHAT_LINK_OFFSET_CLASS,
          )}
        >
          <NewChatIcon />
        </Link>
      </TooltipTrigger>
      <TooltipPortal>
        <TooltipContent side="top">{name}</TooltipContent>
      </TooltipPortal>
    </Tooltip>
  );
}

function NewChatIcon() {
  const { pending } = useLinkStatus();

  if (pending) {
    return <LoadingSpinner size="small" className="!size-4 text-zinc-500" />;
  }

  return <Icon icon={PlusSignIcon} className="size-4" />;
}

function GroupBody({ children }: { children: ReactNode }) {
  return (
    <div className="relative ml-[17px] pl-1.5 before:absolute before:inset-y-0 before:left-0 before:w-px before:bg-gradient-to-b before:from-zinc-200/70 before:to-transparent">
      {children}
    </div>
  );
}

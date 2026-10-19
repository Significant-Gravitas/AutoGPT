"use client";

import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertIdentityDetails } from "@/components/molecules/ExpertIdentityDetails/ExpertIdentityDetails";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { SidebarMenu } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  CheckmarkCircle02Icon,
  PlusSignIcon,
  TickDouble02Icon,
} from "@hugeicons/core-free-icons";
import Link, { useLinkStatus } from "next/link";
import { ReactNode, useState } from "react";
import { EXPERT_CHAT_PAGE_SIZE } from "@/services/experts/expert-chat-pagination";

// The chevron, the slot the header reserves next to it, and the new-chat link
// that floats over that slot are all one size. On touch widths the slot and
// the link grow to a finger-sized target; the chevron keeps its size.
const CHEVRON_SIZE_CLASS = "size-5";
const NEW_CHAT_SLOT_SIZE_CLASS = "size-5 max-md:size-9";
// Where that slot starts, measured in from the header's right edge: the
// trigger's `px-2` (0.5rem) + the chevron's `size-5` (1.25rem) + the `gap-2`
// between them (0.5rem) = 2.25rem. Change any of the three and this moves.
const NEW_CHAT_LINK_OFFSET_CLASS = "right-9";
// The mark-as-read button floats over a second slot, the same size, just left
// of the new-chat one: its offset plus that slot plus the trigger's `gap-2`
// (2.25rem + 1.25rem + 0.5rem = 4rem, or 2.25rem + 2.25rem + 0.5rem = 5rem at
// touch widths). Without a new-chat link it takes that link's place.
const MARK_READ_OFFSET_CLASS = "right-16 max-md:right-20";
// Otto opens with a short preview and a single "Load all"; every other
// expert starts collapsed and pages in when opened.
const OTTO_PREVIEW_COUNT = 1;

interface Props {
  label: string;
  role?: string | null;
  avatarUrl: string | null;
  newChatHref: string | null;
  color?: string | null;
  isAutopilot?: boolean;
  sessions: SessionSummaryResponse[];
  isUnread: (session: SessionSummaryResponse) => boolean;
  onMarkAllRead: () => void;
  renderItem: (session: SessionSummaryResponse) => ReactNode;
}

export function ExpertChatGroup({
  label,
  role,
  avatarUrl,
  newChatHref,
  color,
  isAutopilot,
  sessions,
  isUnread,
  onMarkAllRead,
  renderItem,
}: Props) {
  const [isOpen, setIsOpen] = useState(!!isAutopilot);
  const [visibleCount, setVisibleCount] = useState(
    isAutopilot ? OTTO_PREVIEW_COUNT : EXPERT_CHAT_PAGE_SIZE,
  );
  const visibleSessions = sessions.slice(0, visibleCount);
  const hasHiddenSessions = sessions.length > visibleSessions.length;
  const unreadCount = sessions.filter(isUnread).length;
  // A collapsed group still lists what needs attention: chats running now
  // and chats with a reply the user hasn't opened.
  const attentionSessions = sessions.filter(
    (session) => session.is_processing || isUnread(session),
  );

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
          className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-1 text-left text-sm font-medium text-zinc-900 max-md:py-2 max-md:text-base"
        >
          {isAutopilot ? (
            <AutopilotAvatar
              size={32}
              transparent
              className="rounded-full border border-[#e3e3e3]"
            />
          ) : (
            <ExpertAvatar
              name={label}
              avatarUrl={avatarUrl}
              color={color}
              size={32}
              className="rounded-full border border-[#e3e3e3]"
            />
          )}
          <ExpertIdentityDetails
            isOtto={isAutopilot}
            name={label}
            role={role}
            size="compact"
            areaClassName="opacity-70"
            nameAccessory={
              <>
                <SessionCount count={sessions.length} />
                {unreadCount > 0 && <UnreadCount count={unreadCount} />}
              </>
            }
          />
          {unreadCount > 0 && (
            <span
              aria-hidden
              className={cn("ml-auto shrink-0", NEW_CHAT_SLOT_SIZE_CLASS)}
            />
          )}
          {newChatHref && (
            <span
              aria-hidden
              className={cn(
                "shrink-0",
                unreadCount === 0 && "ml-auto",
                NEW_CHAT_SLOT_SIZE_CLASS,
              )}
            />
          )}
          {/* Hidden at rest on desktop like the new-chat link, and revealed
              by the same header hover / focus-within so keyboard users see
              it the moment the trigger takes focus. Opacity keeps its slot,
              so nothing shifts. On touch widths it stays visible. */}
          <Icon
            icon={ArrowDown01Icon}
            className={cn(
              "ease-[cubic-bezier(0.33,1,0.68,1)] shrink-0 text-zinc-400 transition-[opacity,transform] duration-200 group-focus-within/expert-header:opacity-100 group-hover/expert-header:opacity-100 group-data-[state=open]/expert-group:rotate-180 motion-reduce:transition-none md:opacity-0",
              CHEVRON_SIZE_CLASS,
              !newChatHref && unreadCount === 0 && "ml-auto",
            )}
          />
        </CollapsibleTrigger>
        {unreadCount > 0 && (
          <MarkAllReadButton
            label={label}
            offsetClass={
              newChatHref ? MARK_READ_OFFSET_CLASS : NEW_CHAT_LINK_OFFSET_CLASS
            }
            onClick={onMarkAllRead}
          />
        )}
        {newChatHref && <NewChatLink href={newChatHref} label={label} />}
      </div>

      {!isOpen && attentionSessions.length > 0 && (
        <GroupBody>
          <SidebarMenu>{attentionSessions.map(renderItem)}</SidebarMenu>
        </GroupBody>
      )}

      <CollapsibleContent className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down motion-reduce:animate-none">
        <GroupBody>
          <SidebarMenu>{visibleSessions.map(renderItem)}</SidebarMenu>
          {hasHiddenSessions && (
            <button
              type="button"
              aria-label={`${isAutopilot ? "Load all" : "Load more"} ${label} chats`}
              onClick={() =>
                setVisibleCount((count) =>
                  isAutopilot ? Infinity : count + EXPERT_CHAT_PAGE_SIZE,
                )
              }
              className="mt-0.5 w-full rounded-md px-2 py-1 text-left text-xs font-medium text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800 max-md:py-2.5 max-md:text-sm"
            >
              {isAutopilot ? "Load all" : "Load more"}
            </button>
          )}
        </GroupBody>
      </CollapsibleContent>
    </Collapsible>
  );
}

function SessionCount({ count }: { count: number }) {
  return (
    <span
      aria-hidden
      className="shrink-0 rounded-full bg-zinc-200/70 px-1.5 text-[11px] font-medium tabular-nums leading-[1.125rem] text-zinc-600"
    >
      {count}
    </span>
  );
}

function UnreadCount({ count }: { count: number }) {
  return (
    <span
      aria-hidden
      className="flex shrink-0 items-center gap-0.5 rounded-full bg-green-50 px-1.5 text-[11px] font-medium tabular-nums leading-[1.125rem] text-green-700"
    >
      <Icon icon={CheckmarkCircle02Icon} className="size-3" />
      {count}
    </span>
  );
}

interface MarkAllReadButtonProps {
  label: string;
  offsetClass: string;
  onClick: () => void;
}

function MarkAllReadButton({
  label,
  offsetClass,
  onClick,
}: MarkAllReadButtonProps) {
  const name = `Mark all ${label} chats as read`;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={name}
          onClick={onClick}
          className={cn(
            "absolute top-1/2 flex -translate-y-1/2 items-center justify-center rounded-md text-zinc-500 transition-opacity group-focus-within/expert-header:opacity-100 group-hover/expert-header:opacity-100 hover:bg-zinc-200 hover:text-zinc-900 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 md:opacity-0",
            NEW_CHAT_SLOT_SIZE_CLASS,
            offsetClass,
          )}
        >
          <Icon icon={TickDouble02Icon} className="size-4 max-md:size-5" />
        </button>
      </TooltipTrigger>
      <TooltipPortal>
        <TooltipContent side="top">Mark all as read</TooltipContent>
      </TooltipPortal>
    </Tooltip>
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
            NEW_CHAT_SLOT_SIZE_CLASS,
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

  return <Icon icon={PlusSignIcon} className="size-4 max-md:size-5" />;
}

function GroupBody({ children }: { children: ReactNode }) {
  return (
    <div className="relative ml-[24px] pl-1.5 before:absolute before:inset-y-0 before:left-0 before:w-px before:bg-gradient-to-b before:from-zinc-200/70 before:to-transparent">
      {children}
    </div>
  );
}

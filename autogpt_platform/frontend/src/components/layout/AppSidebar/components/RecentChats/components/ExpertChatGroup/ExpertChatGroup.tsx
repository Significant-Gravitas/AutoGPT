"use client";

import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
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
  CollapsibleRoot,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/molecules/Collapsible/Collapsible";
import { SidebarMenu } from "@/components/molecules/Sidebar/Sidebar";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
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
  renderItem,
}: Props) {
  const [isOpen, setIsOpen] = useState(!!isAutopilot);
  const [visibleCount, setVisibleCount] = useState(
    isAutopilot ? OTTO_PREVIEW_COUNT : EXPERT_CHAT_PAGE_SIZE,
  );
  const visibleSessions = sessions.slice(0, visibleCount);
  const hasHiddenSessions = sessions.length > visibleSessions.length;
  const runningSessions = sessions.filter((session) => session.is_processing);

  return (
    <CollapsibleRoot
      open={isOpen}
      onOpenChange={setIsOpen}
      className="group/expert-group"
    >
      {/* The trigger is itself a button, so the new-chat link sits beside it
          and floats over a slot the trigger reserves before the chevron —
          long labels then truncate against the link rather than under it. */}
      <div className="group/expert-header relative mb-1 flex items-center rounded-md hover:bg-surface-hover">
        <CollapsibleTrigger
          aria-label={`${label} chats`}
          // The size and weight reach ManagedExpertImage's fallback initials (a
          // span that sets its colour, not its size or weight), and the weight
          // ExpertAvatar's initial; every other piece of copy is a toned Text.
          className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-1 text-left text-body font-medium leading-5 max-md:py-2 max-md:text-body-lg max-md:leading-6"
        >
          {isAutopilot ? (
            <AutopilotAvatar
              size={32}
              transparent
              // TODO(tokens): no exact role for border-[#e3e3e3] (#E3E3E3)
              className="rounded-full border border-[#e3e3e3]"
            />
          ) : (
            <ExpertAvatar
              name={label}
              avatarUrl={avatarUrl}
              color={color}
              size={32}
              // TODO(tokens): no exact role for border-[#e3e3e3] (#E3E3E3)
              className="rounded-full border border-[#e3e3e3]"
            />
          )}
          <ExpertIdentityDetails
            isOtto={isAutopilot}
            name={label}
            role={role}
            size="compact"
            areaClassName="opacity-70"
            nameAccessory={<SessionCount count={sessions.length} />}
          />
          {newChatHref && (
            <span
              aria-hidden
              className={cn("ml-auto shrink-0", NEW_CHAT_SLOT_SIZE_CLASS)}
            />
          )}
          {/* Hidden at rest on desktop like the new-chat link, and revealed
              by the same header hover / focus-within so keyboard users see
              it the moment the trigger takes focus. Opacity keeps its slot,
              so nothing shifts. On touch widths it stays visible. */}
          <Icon
            icon={ArrowDown01Icon}
            className={cn(
              "ease-[cubic-bezier(0.33,1,0.68,1)] shrink-0 text-foreground-subtle transition-[opacity,transform] duration-200 group-focus-within/expert-header:opacity-100 group-hover/expert-header:opacity-100 group-data-[state=open]/expert-group:rotate-180 motion-reduce:transition-none md:opacity-0",
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
              aria-label={`${isAutopilot ? "Load all" : "Load more"} ${label} chats`}
              onClick={() =>
                setVisibleCount((count) =>
                  isAutopilot ? Infinity : count + EXPERT_CHAT_PAGE_SIZE,
                )
              }
              className="mt-0.5 w-full rounded-md px-2 py-1 text-left text-foreground-tertiary hover:bg-surface-hover hover:text-foreground-emphasis max-md:py-2.5"
            >
              <Text
                as="span"
                variant="small-medium-tight-touch"
                tone="inherit"
                className="block"
              >
                {isAutopilot ? "Load all" : "Load more"}
              </Text>
            </button>
          )}
        </GroupBody>
      </CollapsibleContent>
    </CollapsibleRoot>
  );
}

function SessionCount({ count }: { count: number }) {
  return (
    <span
      aria-hidden
      className="shrink-0 rounded-full bg-surface-active/70 px-1.5 text-caption-sm font-medium tabular-nums leading-4.5 text-foreground-secondary"
    >
      {count}
    </span>
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
            // TODO(tokens): no fitting role for focus-visible:ring-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour), not a focus ring
            "absolute top-1/2 flex -translate-y-1/2 items-center justify-center rounded-md text-foreground-tertiary transition-opacity hover:bg-surface-active hover:text-foreground-strong focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 group-focus-within/expert-header:opacity-100 group-hover/expert-header:opacity-100 md:opacity-0",
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
    return (
      <LoadingSpinner
        size="small"
        className="!size-4 text-foreground-tertiary"
      />
    );
  }

  return <Icon icon={PlusSignIcon} className="size-4 max-md:size-5" />;
}

function GroupBody({ children }: { children: ReactNode }) {
  return (
    <div className="relative ml-[24px] pl-1.5 before:absolute before:inset-y-0 before:left-0 before:w-px before:bg-gradient-to-b before:from-surface-active/70 before:to-transparent">
      {children}
    </div>
  );
}

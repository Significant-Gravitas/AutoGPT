"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
} from "@/components/ui/sheet";
import { Robot01Icon } from "@hugeicons/core-free-icons";
import { CopilotChatHost } from "../../../copilot/CopilotChatHost";

interface Props {
  open: boolean;
  /** The expert to chat with, or `null` for Autopilot (the generalist). */
  expert: Expert | null;
  onClose: () => void;
}

/**
 * Right-side drawer that hosts the full copilot chat inline on the Team page.
 * The chat is URL-driven (`expertId`/`sessionId`/`kickoff` query params handled
 * in useTeamPage), so mounting the real `CopilotChatHost` here gives faithful
 * streaming/tools/sessions with no forked logic. Radix unmounts the drawer
 * content on close, so each open starts from the freshly-set params.
 */
export function ExpertChatDrawer({ open, expert, onClose }: Props) {
  const title = expert ? `Chat with ${expert.name}` : "Chat with Autopilot";
  const subtitle = expert
    ? expert.role
    : "Your built-in generalist — runs the shop";

  return (
    <Sheet
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
    >
      <SheetContent
        side="right"
        className="flex w-full flex-col overflow-hidden p-0 sm:w-1/2 sm:max-w-none"
      >
        <div className="border-b border-zinc-200 bg-white px-6 py-5 pr-12 sm:px-8">
          <div className="flex items-center gap-3">
            <Avatar className="h-11 w-11">
              {expert?.avatar_url ? (
                <AvatarImage src={expert.avatar_url} alt={expert.name} />
              ) : null}
              <AvatarFallback>
                {expert ? (
                  expert.name
                ) : (
                  <Icon icon={Robot01Icon} size={22} />
                )}
              </AvatarFallback>
            </Avatar>
            <div className="min-w-0">
              <SheetTitle>{title}</SheetTitle>
              <SheetDescription>{subtitle}</SheetDescription>
            </div>
          </div>
        </div>

        {/* Definite-height flex-col parent so CopilotChatHost's `min-h-0
            flex-1` chain lets the messages area absorb growth (SheetContent is
            `inset-y-0 h-full`, giving the viewport-bound height it needs). */}
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <CopilotChatHost
            droppedFiles={[]}
            onDroppedFilesConsumed={() => {}}
            hasFloatingControls={false}
          />
        </div>
      </SheetContent>
    </Sheet>
  );
}

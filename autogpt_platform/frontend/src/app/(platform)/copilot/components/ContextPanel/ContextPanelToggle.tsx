"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { SidebarRightIcon } from "@hugeicons/core-free-icons";
import type { UIMessage } from "ai";
import { useEffect } from "react";
import { useCopilotUIStore, type ContextPanelExpert } from "../../store";
import { useIsMobile } from "../../useIsMobile";
import { useWorkAreaStore } from "../../workArea";

interface Props {
  /** The chat's messages, which the side panel's "This chat" tab reads. */
  messages?: UIMessage[];
  sessionId?: string | null;
  /** The chat's live expert, whose integrations the side panel can open. */
  expert?: ContextPanelExpert | null;
  /** Kept for the callers: the composer's tab shows the integrations on
   *  mobile (new layout, see ComposerPanelTab). */
  hideIntegrationsOnMobile?: boolean;
}

/** The chat's one top-right control: it opens the side panel, or folds it
 *  away. With nothing open the panel comes back on a new tab, which lists
 *  the files, the computer and the expert's integrations. */
export function ContextPanelToggle({
  messages,
  sessionId = null,
  expert = null,
}: Props) {
  const isMobile = useIsMobile();
  const toggle = useWorkAreaStore((s) => s.toggle);
  const isOpen = useWorkAreaStore((s) => s.tabCount > 0 && !s.isFolded);
  const setContextPanelExpert = useCopilotUIStore(
    (s) => s.setContextPanelExpert,
  );

  // The side panel shows what this chat did, read from the same messages the
  // chat renders, whichever engine streams them.
  const publishChat = useWorkAreaStore((s) => s.publishChat);
  useEffect(() => {
    if (sessionId && messages) publishChat(sessionId, messages);
  }, [sessionId, messages, publishChat]);

  // The new tab offers this chat's expert's integrations.
  useEffect(() => {
    setContextPanelExpert(expert);
  }, [expert, setContextPanelExpert]);

  if (!sessionId || isMobile) return null;

  return (
    <div className="mt-3 mr-4 flex shrink-0 items-center">
      <Tooltip inheritDelay>
        <TooltipTrigger asChild>
          <Button
            type="button"
            variant="compact-ghost"
            size="compact-icon"
            onClick={toggle}
            aria-label={isOpen ? "Hide the side panel" : "Open the side panel"}
            aria-pressed={isOpen}
            className="size-8 shrink-0 rounded-lg transition-[background-color,transform] duration-150 ease-out hover:bg-surface-hover active:scale-[0.97] motion-reduce:transition-none"
          >
            <Icon
              icon={SidebarRightIcon}
              tone="emphasis-soft"
              className="!size-4"
            />
          </Button>
        </TooltipTrigger>
        <TooltipPortal>
          <TooltipContent side="bottom">
            {isOpen ? "Hide the side panel" : "Side panel"}
          </TooltipContent>
        </TooltipPortal>
      </Tooltip>
    </div>
  );
}

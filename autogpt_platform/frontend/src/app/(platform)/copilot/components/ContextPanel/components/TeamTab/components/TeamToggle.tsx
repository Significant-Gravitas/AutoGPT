"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { UserGroupIcon } from "@hugeicons/core-free-icons";
import { useCopilotUIStore } from "../../../../../store";
import { useTeamRoster } from "../useTeamRoster";

interface Props {
  sessionId: string | null;
}

/** The team panel's trigger: hidden until the chat delegates to someone,
 *  then badged with how many teammates are still working. The badge drops
 *  while the roster itself is on screen. */
export function TeamToggle({ sessionId }: Props) {
  const { rows, liveCount } = useTeamRoster(sessionId);
  const isOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);
  const activeTab = useCopilotUIStore((s) => s.artifactPanel.activeTab);
  const hasArtifact = useCopilotUIStore(
    (s) => s.artifactPanel.activeArtifact != null,
  );
  const toggleContextPanelTab = useCopilotUIStore(
    (s) => s.toggleContextPanelTab,
  );

  if (rows.length === 0) return null;

  const isTeamOpen = isOpen && activeTab === "team" && !hasArtifact;
  const showBadge = liveCount > 0 && !isTeamOpen;

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      onClick={() => toggleContextPanelTab("team")}
      aria-label={
        isTeamOpen
          ? "Hide team"
          : liveCount > 0
            ? `Open team, ${liveCount} working`
            : "Open team"
      }
      aria-pressed={isTeamOpen}
      className={cn(
        "relative size-8 shrink-0 rounded-md transition-[background-color,transform] duration-150 ease-out hover:bg-zinc-100 active:scale-[0.97] motion-reduce:transition-none",
        isTeamOpen && "bg-zinc-100",
      )}
    >
      <Icon
        icon={UserGroupIcon}
        className="!size-4 text-sidebar-foreground/90"
      />
      {showBadge && (
        <span
          aria-hidden
          className="absolute -right-0.5 -top-0.5 flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-purple-600 px-1 text-[9px] font-semibold tabular-nums text-white"
        >
          {liveCount}
        </span>
      )}
    </Button>
  );
}

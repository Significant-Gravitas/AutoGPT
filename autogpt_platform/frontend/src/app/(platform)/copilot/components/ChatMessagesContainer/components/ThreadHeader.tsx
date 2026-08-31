"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ClockIcon, File02Icon, PlayIcon } from "@hugeicons/core-free-icons";
import { useCopilotUIStore } from "../../../store";
import type { ExpertIdentity } from "../../../useExpertMap";
import { useSessionFiles } from "../../ContextPanel/components/FilesTab/useSessionFiles";
import { useSessionActivity } from "../../WorkspaceFileCards/useSessionActivity";
import { ExpertAvatar } from "./ExpertAvatar/ExpertAvatar";

// Autopilot's product-facing title when a session carries no expert identity.
const DEFAULT_EXPERT_ROLE = "Head of AI";

interface Props {
  expertIdentity?: ExpertIdentity | null;
  readOnly: boolean;
  /** Powers the chip's file/run counters and its click-through to the
   *  session activity card. Without it the chip is a passive label. */
  sessionId?: string | null;
  /** The new layout floats the sidebar and workspace-files controls over the
   *  chat's top-left corner below `lg`; the chip clears them. */
  hasFloatingControls?: boolean;
}

/** Floats as a translucent chip pinned to the pane's top-left gutter — the
 *  zero-height wrapper keeps it out of the flex flow so messages scroll
 *  underneath. On narrow viewports the gutter disappears and the chip simply
 *  overlaps the message column, which its translucency is built for. An
 *  expert session wears the expert's identity and every other session is
 *  Autopilot's, so the thread is never anonymous. Clicking the chip opens
 *  the session activity card (files, runs, schedules) on the right — the
 *  same surface the workspace-files toggle answers with. */
export function ThreadHeader({
  expertIdentity,
  readOnly,
  sessionId = null,
  hasFloatingControls = false,
}: Props) {
  const name = expertIdentity?.name ?? "Autopilot";
  const role = expertIdentity?.role ?? DEFAULT_EXPERT_ROLE;
  const isArtifactsEnabled = useGetFlag(Flag.ARTIFACTS);
  // The activity card only mounts in the copilot chat behind the artifacts
  // flag, so everywhere else (share viewer, builder/memory panels) the chip
  // stays a passive label and fetches nothing.
  const isInteractive = Boolean(isArtifactsEnabled && !readOnly && sessionId);
  const toggleContextPanelTab = useCopilotUIStore(
    (s) => s.toggleContextPanelTab,
  );
  const activitySessionId = isInteractive ? sessionId : null;
  const { runs, schedules } = useSessionActivity(activitySessionId);
  const { uploaded, generated } = useSessionFiles(activitySessionId);
  const fileCount = uploaded.length + generated.length;
  const runCount = runs.length;
  const scheduleCount = schedules.length;

  const chipContent = (
    <>
      <ExpertAvatar
        name={name}
        avatarUrl={expertIdentity?.avatarUrl ?? null}
        isAutopilot={!expertIdentity}
        size="sm"
      />
      <span className="max-w-[10rem] truncate text-sm font-medium text-zinc-800">
        {name}
      </span>
      {isInteractive && fileCount > 0 && (
        <span className="flex shrink-0 items-center gap-1 text-xs font-medium tabular-nums text-zinc-500">
          <Icon icon={File02Icon} className="size-3.5" />
          {fileCount}
        </span>
      )}
      {isInteractive && runCount > 0 && (
        <span className="flex shrink-0 items-center gap-1 text-xs font-medium tabular-nums text-zinc-500">
          <Icon icon={PlayIcon} className="size-3.5" />
          {runCount}
        </span>
      )}
      {isInteractive && scheduleCount > 0 && (
        <span className="flex shrink-0 items-center gap-1 text-xs font-medium tabular-nums text-zinc-500">
          <Icon icon={ClockIcon} className="size-3.5" />
          {scheduleCount}
        </span>
      )}
    </>
  );

  const chipClass =
    "pointer-events-auto flex items-center gap-2 whitespace-nowrap rounded-full border border-zinc-200/70 bg-white/75 py-1 pl-1.5 pr-3 shadow-sm backdrop-blur-md";

  return (
    <div data-testid="expert-thread-header" className="relative z-20 h-0">
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 top-3 flex justify-start px-4",
          hasFloatingControls && "max-md:pl-28 md:max-lg:pl-20",
        )}
      >
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              {isInteractive ? (
                <button
                  type="button"
                  aria-label={`${name} — open session files and runs`}
                  onClick={() => toggleContextPanelTab("files")}
                  className={cn(chipClass, "transition-colors hover:bg-white")}
                >
                  {chipContent}
                </button>
              ) : (
                <div className={chipClass}>{chipContent}</div>
              )}
            </TooltipTrigger>
            <TooltipContent side="bottom">{role}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
}

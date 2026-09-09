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
import { ExpertIntegrations } from "./ExpertIntegrations/ExpertIntegrations";

// Autopilot's product-facing title when a session carries no expert identity.
const DEFAULT_EXPERT_ROLE = "Head of AI";

interface Props {
  expertIdentity?: ExpertIdentity | null;
  readOnly: boolean;
  /** Powers the chip's file/run counters and its click-through to the
   *  session activity card. Without it the chip is a passive label. */
  sessionId?: string | null;
  /** Set only by the host that actually mounts the activity card. The other
   *  hosts pass a real sessionId too, so the flag cannot be inferred — and
   *  without it the chip would open a card that isn't there. */
  canOpenActivity?: boolean;
  /** The new layout floats the sidebar and workspace-files controls over the
   *  chat's top-left corner below `lg`; the chip clears them. */
  hasFloatingControls?: boolean;
}

/** Floats as a translucent chip pinned to the pane's top-left gutter — the
 *  zero-height wrapper keeps it out of the flex flow so messages scroll
 *  underneath. On narrow viewports the gutter disappears and the chip simply
 *  overlaps the message column, which its translucency is built for. An
 *  expert session wears the expert's identity and every other session is
 *  Autopilot's, so the thread is never anonymous. Clicking the identity
 *  opens the session activity card (files, runs, schedules) on the right;
 *  the expert's integration logos sit beside it with their own popover. */
export function ThreadHeader({
  expertIdentity,
  readOnly,
  sessionId = null,
  canOpenActivity = false,
  hasFloatingControls = false,
}: Props) {
  const name = expertIdentity?.name ?? "Autopilot";
  const role = expertIdentity?.role ?? DEFAULT_EXPERT_ROLE;
  const isArtifactsEnabled = useGetFlag(Flag.ARTIFACTS);
  // Only the copilot chat mounts the activity card. The builder and memory
  // panels pass a live sessionId and aren't read-only, so without the host's
  // own say-so the chip would render a button whose click writes panel state
  // and fetches a file list for a card that never appears.
  const isInteractive = Boolean(
    canOpenActivity && isArtifactsEnabled && !readOnly && sessionId,
  );
  const toggleContextPanelTab = useCopilotUIStore(
    (s) => s.toggleContextPanelTab,
  );
  const activitySessionId = isInteractive ? sessionId : null;
  const { runs, schedules } = useSessionActivity(activitySessionId);
  const { uploaded, generated } = useSessionFiles(activitySessionId);
  const fileCount = uploaded.length + generated.length;
  const runCount = runs.length;
  const scheduleCount = schedules.length;
  const showIntegrations = expertIdentity != null && !expertIdentity.isArchived;

  // The chip's own aria-label replaces its contents for screen readers, so
  // every count has to be spelled out here or it is simply never announced.
  const counters = isInteractive
    ? [
        { icon: File02Icon, count: fileCount, noun: "file" },
        { icon: PlayIcon, count: runCount, noun: "run" },
        { icon: ClockIcon, count: scheduleCount, noun: "schedule" },
      ].filter((counter) => counter.count > 0)
    : [];
  const spokenCounts = counters
    .map(({ count, noun }) => `${count} ${noun}${count === 1 ? "" : "s"}`)
    .join(", ");

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
      {counters.map(({ icon, count, noun }) => (
        <span
          key={noun}
          className="flex shrink-0 items-center gap-1 text-xs font-medium tabular-nums text-zinc-500"
        >
          <Icon icon={icon} className="size-3.5" aria-hidden />
          {count}
        </span>
      ))}
    </>
  );

  return (
    <div data-testid="expert-thread-header" className="relative z-20 h-0">
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 top-3 flex justify-start px-4",
          hasFloatingControls && "max-md:pl-28 md:max-lg:pl-20",
        )}
      >
        {/* The integrations popover is its own button, so the chip is a
            container and only the identity half opens the activity card —
            nesting the two triggers would be button-in-button. Each half
            carries its own padding so its hover fill reaches the chip's edge. */}
        <div className="pointer-events-auto flex items-center whitespace-nowrap rounded-full border border-zinc-200/70 bg-white/75 shadow-sm backdrop-blur-md">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                {isInteractive ? (
                  <button
                    type="button"
                    aria-label={
                      spokenCounts
                        ? `${name}, ${role}. ${spokenCounts}. Open session activity`
                        : `${name}, ${role}. Open session activity`
                    }
                    onClick={() => toggleContextPanelTab("files")}
                    className="flex min-w-0 items-center gap-2 rounded-full py-1 pl-1.5 pr-3 transition-colors hover:bg-zinc-100/80"
                  >
                    {chipContent}
                  </button>
                ) : (
                  // The role lives only in the tooltip, and a tooltip opens
                  // on focus as well as hover — so even this passive chip has
                  // to be reachable by keyboard, or read-only viewers never
                  // get the role at all.
                  <div
                    tabIndex={0}
                    aria-label={`${name} — ${role}`}
                    className="flex min-w-0 items-center gap-2 rounded-full py-1 pl-1.5 pr-3"
                  >
                    {chipContent}
                  </div>
                )}
              </TooltipTrigger>
              <TooltipContent
                side="bottom"
                className="bg-zinc-900 text-zinc-50 outline-none"
              >
                {role}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          {showIntegrations && (
            <ExpertIntegrations
              expertId={expertIdentity.id}
              expertName={expertIdentity.name}
            />
          )}
        </div>
      </div>
    </div>
  );
}

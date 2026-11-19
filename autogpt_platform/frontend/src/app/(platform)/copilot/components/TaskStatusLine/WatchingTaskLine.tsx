"use client";

import { RepeatMenu } from "@/app/(platform)/tasks/components/RepeatMenu/RepeatMenu";
import { waitingLine } from "@/app/(platform)/tasks/schedule";
import { useTaskActions } from "@/app/(platform)/tasks/useTaskActions";
import { Icon } from "@/components/atoms/Icon/Icon";
import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";
import { ArrowRight01Icon, PauseIcon } from "@hugeicons/core-free-icons";
import { cn } from "@/lib/utils";
import { useReducedMotion } from "framer-motion";
import { Radar } from "loading-dev";
import { useWorkAreaStore } from "../../workArea";
import type { StartedRepeat, StartedWatch } from "./helpers";

interface Props {
  taskID: string;
  title: string;
  repeat: StartedRepeat;
  watch: StartedWatch;
  /** Why it waits, when something holds or paused it. */
  heldBecause: string | null;
}

/** The box a task started by an event leaves in the chat while it waits:
 *  a radar that sweeps while it listens, its title and what it watches in a few words. The whole box opens the task beside
 *  the chat, where its webhook URL, runs and Pause live; the menu stops it
 *  watching. While a run goes, the normal task line takes its place. */
export function WatchingTaskLine({
  taskID,
  title,
  repeat,
  watch,
  heldBecause,
}: Props) {
  const openTask = useWorkAreaStore((s) => s.openTask);
  const actions = useTaskActions(taskID);
  const isStill = useReducedMotion() ?? false;
  const isBusy = actions.isCancelling || actions.isChangingRepeat;
  const where = heldBecause
    ? `${repeat.paused ? watch.line : waitingLine(repeat)} · ${heldBecause}`
    : waitingLine(repeat);

  return (
    <div
      data-testid="watching-task-line"
      className="flex min-h-[3.25rem] items-center gap-1 rounded-card bg-surface pr-2 shadow-(--inbox-shadow-card) transition-shadow duration-150 ease-out hover:shadow-(--inbox-shadow-card-hover)"
    >
      <button
        type="button"
        onClick={() => openTask(taskID, title)}
        className="group/watch flex min-w-0 flex-1 items-center gap-3 rounded-card py-2 pr-1 pl-3 text-left outline-none focus-visible:ring-2 focus-visible:ring-focus-inbox"
      >
        <span
          aria-hidden
          className={cn(
            "flex size-[1.125rem] shrink-0 items-center justify-center",
            repeat.paused ? "text-foreground-quiet" : "text-info",
          )}
        >
          {repeat.paused ? (
            <Icon icon={PauseIcon} size={18} tone="inherit" />
          ) : (
            <Radar size={18} playState={isStill ? "paused" : "running"} />
          )}
        </span>
        <span className="flex min-w-0 flex-1 flex-col">
          <OverflowText
            value={title}
            overflow="fade"
            variant="inbox-medium"
            tone="ink"
          />
          <OverflowText
            value={where}
            overflow="fade"
            variant="inbox"
            tone="quiet"
          />
        </span>
        <span className="flex shrink-0 items-center gap-0.5 text-body-sm text-foreground-quiet transition-colors duration-150 group-hover/watch:text-foreground-ink">
          Details
          <span className="transition-transform duration-150 ease-out group-hover/watch:translate-x-0.5 motion-reduce:transition-none">
            <Icon icon={ArrowRight01Icon} size={14} tone="inherit" />
          </span>
        </span>
      </button>
      <RepeatMenu
        title={title}
        isBusy={isBusy}
        isWatching
        onStopRepeating={actions.stopRepeating}
        onCancel={actions.cancel}
      />
    </div>
  );
}

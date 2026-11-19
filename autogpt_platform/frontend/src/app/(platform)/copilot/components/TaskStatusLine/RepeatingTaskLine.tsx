"use client";

import { RepeatMenu } from "@/app/(platform)/tasks/components/RepeatMenu/RepeatMenu";
import { waitingLine } from "@/app/(platform)/tasks/schedule";
import { useStartCountdown } from "@/app/(platform)/tasks/useStartCountdown";
import { useTaskActions } from "@/app/(platform)/tasks/useTaskActions";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { PauseIcon, PlayIcon, RepeatIcon } from "@hugeicons/core-free-icons";
import type { StartedRepeat } from "./helpers";

interface Props {
  taskID: string;
  title: string;
  owner: { name: string; avatarUrl: string | null };
  repeat: StartedRepeat;
  startAt: Date | string | null;
  /** Why it waits, when something holds or paused it. */
  heldBecause: string | null;
}

/** The box a repeating task leaves in the chat between runs: how often it
 *  runs and when next ("Every day at 08:00 · next run tomorrow 8:00 AM"),
 *  or Paused, with Run now, Pause or Resume, and a menu to stop it
 *  repeating. A task started by an event has its own box
 *  (`WatchingTaskLine`). While a run goes, the normal task line takes its
 *  place. */
export function RepeatingTaskLine({
  taskID,
  title,
  owner,
  repeat,
  startAt,
  heldBecause,
}: Props) {
  const countdown = useStartCountdown(repeat.paused ? null : startAt);
  const actions = useTaskActions(taskID);
  const isBusy =
    actions.isStarting || actions.isCancelling || actions.isChangingRepeat;
  const rhythm = waitingLine(repeat, countdown?.label);
  const where = heldBecause
    ? repeat.paused
      ? `${repeat.label} · ${heldBecause}`
      : `${rhythm} · ${heldBecause}`
    : rhythm;

  return (
    <div
      data-testid="repeating-task-line"
      className="flex min-h-[3.25rem] items-center gap-3 rounded-card bg-surface py-2 pr-2 pl-3 shadow-(--inbox-shadow-card)"
    >
      <span aria-hidden className="flex size-[1.125rem] shrink-0 items-center">
        <Icon
          icon={repeat.paused ? PauseIcon : RepeatIcon}
          size={18}
          tone={repeat.paused ? "quiet" : "info"}
        />
      </span>
      <div className="flex min-w-0 flex-1 flex-col">
        <OverflowText
          value={title}
          overflow="fade"
          variant="inbox-medium"
          tone="ink"
        />
        <span className="flex min-w-0 items-center gap-1.5">
          <span aria-hidden className="inline-flex shrink-0">
            <ExpertAvatar
              name={owner.name}
              avatarUrl={owner.avatarUrl}
              size={16}
              shape="soft"
            />
          </span>
          <Text as="span" variant="inbox" tone="text" className="shrink-0">
            {owner.name}
          </Text>
          <Text as="span" variant="inbox" tone="faint" aria-hidden>
            ·
          </Text>
          <OverflowText
            value={where}
            overflow="fade"
            variant="inbox"
            tone="quiet"
            tabular
            className="min-w-0 flex-1"
          />
        </span>
      </div>
      <Button
        variant="primary"
        size="xs"
        leadingIcon={PlayIcon}
        loading={actions.isStarting}
        disabled={isBusy}
        onClick={() => actions.run()}
      >
        Run now
      </Button>
      <Button
        variant="secondary"
        size="xs"
        disabled={isBusy}
        onClick={repeat.paused ? actions.resume : actions.pause}
      >
        {repeat.paused ? "Resume" : "Pause"}
      </Button>
      <RepeatMenu
        title={title}
        isBusy={isBusy}
        onStopRepeating={actions.stopRepeating}
        onCancel={actions.cancel}
      />
    </div>
  );
}

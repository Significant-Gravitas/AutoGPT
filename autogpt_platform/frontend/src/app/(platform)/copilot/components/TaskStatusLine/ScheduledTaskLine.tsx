"use client";

import { useStartCountdown } from "@/app/(platform)/tasks/useStartCountdown";
import { useTaskActions } from "@/app/(platform)/tasks/useTaskActions";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";
import { isRightToLeft } from "../WorkArea/textDirection";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { Clock01Icon, PlayIcon } from "@hugeicons/core-free-icons";

interface Props {
  taskID: string;
  title: string;
  owner: { name: string; avatarUrl: string | null };
  startAt: Date | string | null;
  isMissed: boolean;
  /** Why it has not started, when something holds it. */
  heldBecause: string | null;
}

/** The box a task scheduled for later leaves in the chat: its title, who
 *  will do it, a live "Starts in …" with the exact time, and Start now or
 *  Cancel. Once it starts, the normal task line takes its place. */
export function ScheduledTaskLine({
  taskID,
  title,
  owner,
  startAt,
  isMissed,
  heldBecause,
}: Props) {
  const countdown = useStartCountdown(isMissed ? null : startAt);
  const actions = useTaskActions(taskID);
  const isBusy = actions.isStarting || actions.isCancelling;
  const when = isMissed
    ? "Missed its start time"
    : countdown
      ? `${countdown.label} · ${countdown.exact}`
      : "Scheduled";
  const where = heldBecause && !isMissed ? `${when} · ${heldBecause}` : when;

  return (
    <div
      data-testid="scheduled-task-line"
      className="flex min-h-[3.25rem] items-center gap-3 rounded-card bg-surface py-2 pr-2 pl-3 shadow-(--inbox-shadow-card)"
    >
      <span aria-hidden className="flex size-[1.125rem] shrink-0 items-center">
        <Icon
          icon={Clock01Icon}
          size={18}
          tone={isMissed ? "waiting" : "info"}
        />
      </span>
      <div className="flex min-w-0 flex-1 flex-col">
        <OverflowText
          value={title}
          dir="auto"
          overflow={isRightToLeft(title) ? "ellipsis" : "fade"}
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
            tone={isMissed ? "waiting" : "quiet"}
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
        Start now
      </Button>
      <Button
        variant="secondary"
        size="xs"
        loading={actions.isCancelling}
        disabled={isBusy}
        onClick={actions.cancel}
      >
        Cancel
      </Button>
    </div>
  );
}

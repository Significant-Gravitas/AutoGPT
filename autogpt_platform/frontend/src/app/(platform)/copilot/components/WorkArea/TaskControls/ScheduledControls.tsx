"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Clock01Icon, PlayIcon } from "@hugeicons/core-free-icons";

interface Props {
  isMissed: boolean;
  isStarting: boolean;
  isCancelling: boolean;
  onStartNow: () => void;
  onCancel: () => void;
}

/** The header of a task waiting for its start time, or one that missed it:
 *  start it now, or cancel it before anything runs. */
export function ScheduledControls({
  isMissed,
  isStarting,
  isCancelling,
  onStartNow,
  onCancel,
}: Props) {
  const isBusy = isStarting || isCancelling;
  return (
    <div className="flex flex-col gap-2">
      {isMissed ? (
        <p role="status" className="flex items-center gap-1.5">
          <Icon icon={Clock01Icon} size={14} tone="waiting" />
          <Text as="span" variant="inbox" tone="quiet">
            Missed its start time
          </Text>
        </p>
      ) : null}
      <div className="flex items-center gap-2">
        <Button
          variant="primary"
          size="xs"
          leadingIcon={PlayIcon}
          loading={isStarting}
          disabled={isBusy}
          onClick={onStartNow}
        >
          Start now
        </Button>
        <Button
          variant="secondary"
          size="xs"
          loading={isCancelling}
          disabled={isBusy}
          onClick={onCancel}
        >
          Cancel
        </Button>
      </div>
    </div>
  );
}

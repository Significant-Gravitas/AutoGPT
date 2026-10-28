"use client";

import { useReadEveryRender } from "@/hooks/useReadEveryRender";
import type { AtResetTurn } from "@/app/api/__generated__/models/atResetTurn";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Clock01Icon } from "@hugeicons/core-free-icons";
import { scheduledSendLabel } from "./helpers";
import { useScheduledSendDue } from "./useScheduledSendDue";

interface Props {
  job: AtResetTurn;
  onEdit: () => void;
  onCancel: () => void;
  isBusy?: boolean;
}

export function ScheduledSendChip({ job, onEdit, onCancel, isBusy }: Props) {
  // Once due it may already have gone: editing it then could send it twice.
  const isDue = useScheduledSendDue(job.run_at);
  // Its wording turns on the day, so read on every render rather than cached.
  const label = useReadEveryRender(() => scheduledSendLabel(job.run_at));
  return (
    <div
      role="status"
      aria-label="Scheduled message"
      // TODO(tokens): no exact role for border-purple-200, bg-purple-50
      className="mb-2 flex items-center justify-between gap-3 rounded-2xl border border-purple-200 bg-purple-50 px-3 py-2"
    >
      <div className="flex min-w-0 items-center gap-2">
        <Icon
          icon={Clock01Icon}
          size={16}
          // TODO(tokens): no fitting role for text-purple-600 #6C2EDF; #6C2EDF is brand-hover (a fill)
          className="shrink-0 text-purple-600"
        />
        <Text
          // TODO(tokens): no exact role for !text-purple-900
          variant="small-medium"
          className="shrink-0 !text-purple-900"
        >
          {isDue ? "Sending…" : label}
        </Text>
        {job.message && (
          <Text
            variant="small"
            className="min-w-0 truncate !text-foreground-tertiary"
          >
            {job.message}
          </Text>
        )}
      </div>
      {!isDue && (
        <div className="flex shrink-0 items-center gap-1">
          <Button size="xs" variant="ghost" onClick={onEdit} disabled={isBusy}>
            Edit
          </Button>
          <Button
            size="xs"
            variant="ghost"
            onClick={onCancel}
            disabled={isBusy}
          >
            Cancel
          </Button>
        </div>
      )}
    </div>
  );
}

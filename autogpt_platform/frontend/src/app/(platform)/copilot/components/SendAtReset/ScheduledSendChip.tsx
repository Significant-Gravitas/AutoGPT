"use client";

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
  return (
    <div
      role="status"
      aria-label="Scheduled message"
      className="mb-2 flex items-center justify-between gap-3 rounded-2xl border border-purple-200 bg-purple-50 px-3 py-2"
    >
      <div className="flex min-w-0 items-center gap-2">
        <Icon
          icon={Clock01Icon}
          size={16}
          className="shrink-0 text-purple-600"
        />
        <Text variant="small-medium" className="shrink-0 !text-purple-900">
          {isDue ? "Sending…" : scheduledSendLabel(job.run_at)}
        </Text>
        {job.message && (
          <Text variant="small" className="min-w-0 truncate !text-zinc-500">
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

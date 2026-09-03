"use client";

import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { cn } from "@/lib/utils";
import Link from "next/link";
import type { ReactNode } from "react";
import { useGraphScheduleListItem } from "./useGraphScheduleListItem";
import {
  Calendar03Icon,
  Delete02Icon,
  EyeIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  schedule: GraphExecutionJobInfo;
  /** Extra action rendered before View/Delete — lets callers that know the
   *  schedule's graph plug in an edit affordance without coupling this
   *  shared row to a feature-specific modal. */
  editAction?: ReactNode;
  /** Layout overrides for narrow containers (e.g. drawers) where the
   *  side-by-side sm layout would crush the text column. */
  className?: string;
}

export function GraphScheduleListItem({
  schedule,
  editAction,
  className,
}: Props) {
  const {
    nextRunLabel,
    nextRunRelative,
    nextRunTitle,
    recurrenceLabel,
    agentLabel,
    agentHref,
    isDeleteOpen,
    openDelete,
    closeDelete,
    isDeleting,
    handleDelete,
    isViewOpen,
    openView,
    closeView,
  } = useGraphScheduleListItem({ schedule });

  return (
    <div
      className={cn(
        "flex w-full flex-col gap-3 rounded-large border border-zinc-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between",
        className,
      )}
      data-testid="schedule-row"
      data-schedule-id={schedule.id}
      data-schedule-kind="graph"
    >
      <Link
        href={agentHref}
        className="flex min-w-0 flex-1 items-start gap-3 hover:opacity-80"
        data-testid="schedule-open-agent"
      >
        <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-large border border-slate-50 bg-emerald-50">
          <Icon icon={Calendar03Icon} size={18} className="text-emerald-700" />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <Text
            variant="body-medium"
            className="block w-full truncate text-ellipsis"
          >
            {agentLabel}
          </Text>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
            <Text
              variant="small"
              className="!text-zinc-500"
              title={nextRunTitle}
            >
              {nextRunLabel}
            </Text>
            <span className="text-zinc-300">•</span>
            <Text variant="small" className="!text-zinc-500">
              {recurrenceLabel}
            </Text>
            <span className="text-zinc-300">•</span>
            <span
              className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700"
              data-testid="schedule-kind-badge"
            >
              Agent run
            </span>
          </div>
        </div>
      </Link>

      <div className="flex flex-shrink-0 items-center gap-2">
        {editAction}
        <Button
          variant="secondary"
          size="small"
          onClick={openView}
          data-testid="schedule-view-button"
          aria-label="View schedule"
        >
          <Icon icon={EyeIcon} className="mr-1 h-4 w-4" />
          View
        </Button>
        <Button
          variant="secondary"
          size="small"
          onClick={openDelete}
          data-testid="schedule-delete-button"
          aria-label="Delete schedule"
        >
          <Icon icon={Delete02Icon} className="mr-1 h-4 w-4" />
          Delete
        </Button>
      </div>

      <Dialog
        controlled={{ isOpen: isViewOpen, set: closeView }}
        styling={{ maxWidth: "26rem" }}
        title="Scheduled agent run"
      >
        <Dialog.Content>
          <div className="flex flex-col gap-5">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl border border-emerald-100 bg-emerald-50">
                <Icon
                  icon={Calendar03Icon}
                  size={20}
                  className="text-emerald-700"
                />
              </div>
              <div className="min-w-0">
                <Text variant="body-medium" className="truncate">
                  {agentLabel}
                </Text>
                <Text variant="small" className="!text-zinc-500">
                  Recurring agent run
                </Text>
              </div>
            </div>
            <dl className="divide-y divide-zinc-100 overflow-hidden rounded-xl border border-zinc-200/80 bg-zinc-50/60">
              <ScheduleMetaRow
                label="Next run"
                value={nextRunRelative ?? "Pending"}
                title={nextRunTitle}
              />
              <ScheduleMetaRow label="Repeats" value={recurrenceLabel} />
              <ScheduleMetaRow label="Schedule name" value={schedule.name} />
              <ScheduleMetaRow
                label="Graph"
                value={`${schedule.graph_id} · v${schedule.graph_version}`}
                mono
              />
            </dl>
          </div>
        </Dialog.Content>
      </Dialog>

      <Dialog
        controlled={{ isOpen: isDeleteOpen, set: closeDelete }}
        styling={{ maxWidth: "32rem" }}
        title="Delete scheduled agent run"
      >
        <Dialog.Content>
          <div className="flex flex-col gap-4">
            <Text variant="large">
              Delete this scheduled agent run? The agent will stop running on
              this schedule and you can recreate it from the builder if needed.
            </Text>
            <Dialog.Footer>
              <Button
                variant="secondary"
                disabled={isDeleting}
                onClick={() => closeDelete(false)}
              >
                Keep it
              </Button>
              <Button
                variant="destructive"
                onClick={handleDelete}
                loading={isDeleting}
                data-testid="schedule-confirm-delete"
              >
                Yes, delete
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}

function ScheduleMetaRow({
  label,
  value,
  title,
  mono,
}: {
  label: string;
  value: string;
  title?: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-6 px-4 py-2.5">
      <dt className="flex-shrink-0 text-[13px] text-zinc-500">{label}</dt>
      <dd
        className={cn(
          "min-w-0 truncate text-right text-[13px] font-medium text-zinc-800",
          mono && "font-mono text-xs font-normal tracking-tight text-zinc-600",
        )}
        title={title ?? value}
      >
        {value}
      </dd>
    </div>
  );
}

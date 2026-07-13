"use client";

import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { CalendarDotsIcon, EyeIcon, TrashIcon } from "@phosphor-icons/react";
import Link from "next/link";
import { useGraphScheduleListItem } from "./useGraphScheduleListItem";

interface Props {
  schedule: GraphExecutionJobInfo;
}

export function GraphScheduleListItem({ schedule }: Props) {
  const {
    nextRunLabel,
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
      className="flex w-full flex-col gap-3 rounded-large border border-zinc-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between"
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
          <CalendarDotsIcon
            size={18}
            className="text-emerald-700"
            weight="bold"
          />
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
        <Button
          variant="secondary"
          size="small"
          onClick={openView}
          data-testid="schedule-view-button"
          aria-label="View schedule"
        >
          <EyeIcon className="mr-1 h-4 w-4" />
          View
        </Button>
        <Button
          variant="secondary"
          size="small"
          onClick={openDelete}
          data-testid="schedule-delete-button"
          aria-label="Delete schedule"
        >
          <TrashIcon className="mr-1 h-4 w-4" />
          Delete
        </Button>
      </div>

      <Dialog
        controlled={{ isOpen: isViewOpen, set: closeView }}
        styling={{ maxWidth: "40rem" }}
        title="Scheduled agent run"
      >
        <Dialog.Content>
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
              <Text variant="small" className="!text-zinc-500">
                {nextRunLabel}
              </Text>
              <span className="text-zinc-300">•</span>
              <Text variant="small" className="!text-zinc-500">
                {recurrenceLabel}
              </Text>
              <span className="text-zinc-300">•</span>
              <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700">
                Agent run
              </span>
            </div>
            <Text variant="body-medium">{agentLabel}</Text>
            <Text variant="small" className="!text-zinc-500">
              Schedule name: {schedule.name}
            </Text>
            <Text variant="small" className="!text-zinc-500">
              Graph: {schedule.graph_id} (v{schedule.graph_version})
            </Text>
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

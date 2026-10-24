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
  /** Tint override for the leading icon tile, e.g. an expert's accent. */
  iconClassName?: string;
  /** Size override for the View/Delete buttons, e.g. compact row actions. */
  actionClassName?: string;
  /** `body-md` inside a profile tab, whose copy reads at 15px. */
  textSize?: "body" | "body-md";
}

export function GraphScheduleListItem({
  schedule,
  editAction,
  className,
  iconClassName,
  actionClassName,
  textSize = "body",
}: Props) {
  const {
    nextRunLabel,
    nextRunValue,
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
        "flex w-full flex-col gap-3 rounded-lg border border-border-base bg-surface p-3 sm:flex-row sm:items-center sm:justify-between",
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
        <div
          className={cn(
            // TODO(tokens): no exact role for border-slate-50 #F7F8F9, bg-emerald-50 #ecfdf5
            "flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-md border border-slate-50 bg-emerald-50",
            iconClassName,
          )}
        >
          <Icon
            icon={Calendar03Icon}
            size={18}
            // TODO(tokens): no exact role for text-emerald-700 #047857
            className={iconClassName ? "text-inherit" : "text-emerald-700"}
          />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <Text
            variant={textSize === "body-md" ? "body-md-medium" : "body-medium"}
            className="block w-full truncate text-ellipsis"
          >
            {agentLabel}
          </Text>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
            <Text
              variant="small"
              className="!text-foreground-tertiary"
              title={nextRunTitle}
            >
              {nextRunLabel}
            </Text>
            <span className="text-foreground-disabled">•</span>
            <Text variant="small" className="!text-foreground-tertiary">
              {recurrenceLabel}
            </Text>
            <span className="text-foreground-disabled">•</span>
            <span
              // TODO(tokens): no exact role for bg-emerald-100 #d1fae5
              className="flex rounded-md bg-emerald-100 px-2 py-0.5"
              data-testid="schedule-kind-badge"
            >
              <Text variant="small-tight" as="span" tone="success-strong">
                Workflow run
              </Text>
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
          className={actionClassName}
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
          className={actionClassName}
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
        title="Scheduled workflow run"
      >
        <Dialog.Content>
          <div className="flex flex-col gap-5">
            <div className="flex items-center gap-3">
              <div
                // TODO(tokens): no exact role for border-emerald-100 #d1fae5, bg-emerald-50 #ecfdf5
                className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl border border-emerald-100 bg-emerald-50"
              >
                <Icon
                  icon={Calendar03Icon}
                  size={20}
                  // TODO(tokens): no exact role for text-emerald-700 #047857
                  className="text-emerald-700"
                />
              </div>
              <div className="min-w-0">
                <Text variant="body-medium" className="truncate">
                  {agentLabel}
                </Text>
                <Text variant="small" className="!text-foreground-tertiary">
                  Recurring workflow run
                </Text>
              </div>
            </div>
            <dl className="divide-y divide-separator overflow-hidden rounded-xl border border-border-base/80 bg-surface-muted/60">
              <ScheduleMetaRow
                label="Next run"
                value={nextRunValue}
                title={nextRunTitle}
              />
              <ScheduleMetaRow label="Repeats" value={recurrenceLabel} />
              <ScheduleMetaRow label="Schedule name" value={schedule.name} />
              <ScheduleMetaRow
                label="Workflow"
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
        title="Delete scheduled workflow run"
      >
        <Dialog.Content>
          <div className="flex flex-col gap-4">
            <Text variant="large">
              Delete this scheduled workflow run? The workflow will stop running
              on this schedule and you can recreate it from the builder if
              needed.
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
      <dt className="flex-shrink-0 text-body-sm leading-inherit text-foreground-tertiary">
        {label}
      </dt>
      <Text
        as="dd"
        variant={mono ? "small-mono-tight-tracking" : "body-sm-medium-inherit"}
        tone={mono ? "secondary" : "emphasis"}
        className="min-w-0 truncate text-right"
        title={title ?? value}
      >
        {value}
      </Text>
    </div>
  );
}

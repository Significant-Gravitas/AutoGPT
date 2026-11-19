"use client";

import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { TimeInput } from "@/components/atoms/TimeInput/TimeInput";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";
import { Calendar03Icon, Delete02Icon } from "@hugeicons/core-free-icons";
import { BlockNote, BlockRow } from "../../blocks";
import { QUICK_SCHEDULES } from "../schedule";
import { useSchedulePanel } from "./useSchedulePanel";

interface Props {
  graphID: string;
  agent: LibraryAgent | undefined;
  run: GraphExecutionMeta | undefined;
  onClose: () => void;
}

/** Repeat the workflow: a few one-click times, a time of day to pick, and
 *  the schedules it already has, each with a way to remove it. */
export function SchedulePanel({ graphID, agent, run, onClose }: Props) {
  const panel = useSchedulePanel({ graphID, agent, run });

  return (
    <BlockNote className="gap-3">
      <div
        role="group"
        aria-label="Schedule this workflow"
        className="flex flex-col gap-2.5"
      >
        <div className="flex flex-wrap gap-1.5">
          {QUICK_SCHEDULES.map((choice) => (
            <Button
              key={choice.label}
              variant="secondary"
              size="xs"
              disabled={panel.isSaving}
              onClick={() => panel.add(choice.cron, choice.label.toLowerCase())}
            >
              {choice.label}
            </Button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <Text variant="small" as="span" tone="quiet" className="shrink-0">
            Every day at
          </Text>
          {/* The field keeps room for an error below it; this row has its own. */}
          <div className="h-9 min-w-0">
            <TimeInput
              size="small"
              aria-label="Time of day"
              value={panel.time}
              onChange={panel.setTime}
              wrapperClassName="min-w-0"
            />
          </div>
          <Button
            variant="primary"
            size="xs"
            loading={panel.isSaving}
            onClick={panel.addAtTime}
          >
            Add
          </Button>
        </div>
        {panel.error ? (
          <Text as="p" variant="small" tone="error" role="alert">
            {panel.error}
          </Text>
        ) : null}
      </div>
      {panel.schedules.length > 0 ? (
        <ul aria-label="Schedules" className="-mx-2 flex flex-col">
          {panel.schedules.map((schedule) => (
            <BlockRow
              key={schedule.id}
              tile={<Icon icon={Calendar03Icon} size={14} tone="inherit" />}
              title={safeHumanizeCronExpression(schedule.cron)}
              subtitle={`Next run ${new Date(schedule.next_run_time).toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" })}`}
              trailing={
                <Button
                  variant="ghost-danger"
                  size="xs"
                  leadingIcon={Delete02Icon}
                  loading={panel.removingID === schedule.id}
                  aria-label={`Remove ${safeHumanizeCronExpression(schedule.cron)}`}
                  onClick={() => panel.remove(schedule.id)}
                >
                  Remove
                </Button>
              }
            />
          ))}
        </ul>
      ) : (
        <Text variant="small" as="p" tone="quiet">
          No schedules yet.
        </Text>
      )}
      <Button
        variant="ghost-muted"
        size="xs"
        className="self-end"
        onClick={onClose}
      >
        Close
      </Button>
    </BlockNote>
  );
}

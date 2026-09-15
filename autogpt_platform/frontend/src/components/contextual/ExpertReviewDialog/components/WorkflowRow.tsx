import type { WorkflowResolution } from "@/app/api/__generated__/models/workflowResolution";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";
import { cn } from "@/lib/utils";
import { getWorkflowSourceLabel } from "../helpers";

interface Props {
  workflow: WorkflowResolution;
  isRemoved: boolean;
  isScheduled: boolean;
  onToggle: () => void;
  onToggleSchedule: () => void;
  readOnly?: boolean;
}

export function WorkflowRow({
  workflow,
  isRemoved,
  isScheduled,
  onToggle,
  onToggleSchedule,
  readOnly,
}: Props) {
  const source = getWorkflowSourceLabel(workflow.source);
  const scheduleId = `review-schedule-${workflow.index}`;

  return (
    <li
      className={cn(
        "flex flex-col gap-2 rounded-lg px-3 py-2 ring-1 ring-inset ring-zinc-200",
        isRemoved && "opacity-50",
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <Text variant="body-medium" className="truncate text-zinc-900">
            {workflow.name}
          </Text>
          <Badge variant={source.variant} size="small">
            {source.label}
          </Badge>
        </div>
        {readOnly ? null : (
          <Button
            variant="ghost"
            size="xs"
            onClick={onToggle}
            aria-label={`${isRemoved ? "Keep" : "Remove"} ${workflow.name}`}
          >
            {isRemoved ? "Undo" : "Remove"}
          </Button>
        )}
      </div>

      {workflow.schedule_cron && !isRemoved ? (
        <div className="flex items-center justify-between gap-3">
          {readOnly ? (
            <Text variant="small" className="text-zinc-600">
              {`Schedule · ${safeHumanizeCronExpression(workflow.schedule_cron)}`}
            </Text>
          ) : (
            <>
              <label htmlFor={scheduleId} className="text-sm text-zinc-600">
                Schedule
                <span className="block text-xs text-zinc-500">
                  {safeHumanizeCronExpression(workflow.schedule_cron)}
                </span>
              </label>
              <Switch
                id={scheduleId}
                checked={isScheduled}
                onCheckedChange={onToggleSchedule}
              />
            </>
          )}
        </div>
      ) : null}
    </li>
  );
}

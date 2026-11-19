import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { Task } from "@/app/api/__generated__/models/task";
import { runHref } from "@/app/(platform)/runs/helpers";
import { BOARD_HREF, taskHref } from "@/app/(platform)/tasks/helpers";

import type { PanelAction } from "../../affect/agent-schedules";

export interface ActionArgs {
  row: string;
  task: Task | undefined;
  workflow: GraphExecutionJobInfo | undefined;
  executions: GraphExecutionMeta[];
  confirming: string | null;
  setConfirming: (row: string | null) => void;
  onRemove: (row: string) => void;
  onOpen: (href: string) => void;
}

/** The panel's own buttons after Run now and Pause: where the schedule
 *  lives, and the one that ends it, which asks once more before it acts. */
export function panelActions(args: ActionArgs): PanelAction[] {
  const open = openAction(args);
  const end = endWords(args);
  if (!end) return open;
  if (args.confirming !== args.row) {
    return [
      ...open,
      {
        label: end.first,
        danger: true,
        onClick: () => args.setConfirming(args.row),
      },
    ];
  }
  return [
    ...open,
    {
      label: end.confirm,
      danger: true,
      onClick: () => {
        args.setConfirming(null);
        args.onRemove(args.row);
      },
    },
    { label: "Keep", onClick: () => args.setConfirming(null) },
  ];
}

function openAction({ task, workflow, executions, onOpen }: ActionArgs) {
  if (task) {
    const href = taskHref(task) ?? BOARD_HREF;
    return [{ label: "Open task", onClick: () => onOpen(href) }];
  }
  const last = executions[0];
  if (!workflow || !last) return [];
  return [
    {
      label: "Open last run",
      onClick: () => onOpen(runHref(workflow.graph_id, last.id)),
    },
  ];
}

/** The two buttons of an armed ending, which keep it armed while focused. */
export function armedLabels(args: Pick<ActionArgs, "task" | "workflow">) {
  const end = endWords(args);
  return end ? [end.confirm, "Keep"] : [];
}

/** How the schedule ends: a workflow schedule is deleted, a repeating or
 *  watching task stops, a task waiting for one start is cancelled. */
function endWords({ task, workflow }: Pick<ActionArgs, "task" | "workflow">) {
  if (workflow) return { first: "Delete", confirm: "Delete this schedule" };
  if (!task) return null;
  if (task.repeat?.watch) {
    return { first: "Stop watching", confirm: "Stop watching for good" };
  }
  if (task.repeat) {
    return { first: "Stop repeating", confirm: "Stop repeating for good" };
  }
  return { first: "Cancel", confirm: "Cancel this task" };
}

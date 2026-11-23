import { toast as sonnerToast } from "sonner";

import { BOARD_HREF, taskHref } from "@/app/(platform)/tasks/helpers";
import type { Task } from "@/app/api/__generated__/models/task";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";

import { runHref } from "../runs/helpers";

const STARTED_KEY = "schedules-started";
const REMOVED_KEY = "schedules-removed";

/** Paused, resumed, or put back. */
export function switchedToast(title: string) {
  toast({ title, variant: "success" });
}

/** Stopped repeating, cancelled or deleted; a deleted workflow schedule
 *  offers Undo. */
export function removedToast(title: string, onUndo?: () => void) {
  toast({
    title,
    variant: "success",
    replaceKey: REMOVED_KEY,
    duration: onUndo ? 8000 : 5000,
    action: onUndo ? (
      <Button
        size="small"
        variant="secondary"
        onClick={() => {
          sonnerToast.dismiss(REMOVED_KEY);
          onUndo();
        }}
      >
        Undo
      </Button>
    ) : undefined,
  });
}

interface Started {
  task?: Task;
  run?: { graphID: string; runID: string };
}

/** Run now worked: where the run went – the task, or the workflow run's
 *  own page. */
export function startedToast(name: string, { task, run }: Started) {
  const href = run
    ? runHref(run.graphID, run.runID)
    : task
      ? (taskHref(task) ?? BOARD_HREF)
      : null;
  toast({
    title: `Started ${name}`,
    variant: "success",
    replaceKey: STARTED_KEY,
    action: href ? (
      <Button
        as="NextLink"
        href={href}
        size="small"
        variant="secondary"
        onClick={() => sonnerToast.dismiss(STARTED_KEY)}
      >
        {run ? "Open run" : "Open task"}
      </Button>
    ) : undefined,
  });
}

export function failedToast(title: string, error: unknown) {
  toast({
    title,
    description: error instanceof Error ? error.message : undefined,
    variant: "destructive",
  });
}

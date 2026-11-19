import type { TaskUpdateRequest } from "@/app/api/__generated__/models/taskUpdateRequest";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";

/** Long enough to read the toast and reach its button. */
const ACTION_TOAST_MS = 8_000;

export function showError(title: string, error: unknown) {
  toast({
    title,
    description: error instanceof Error ? error.message : undefined,
    variant: "destructive",
  });
}

export function updateErrorTitle(data: TaskUpdateRequest) {
  if ("start_at" in data) {
    return data.start_at
      ? "Couldn't change the start time"
      : "Couldn't remove the start time";
  }
  return data.title ? "Couldn't rename the task" : "Couldn't change the status";
}

/** A change that never left the browser: nothing happened, try again. */
export function showUnreachable(onRetry: () => void) {
  toast({
    title: "Couldn't reach AutoGPT",
    description: "Nothing was changed. Check your connection and try again.",
    variant: "destructive",
    duration: ACTION_TOAST_MS,
    action: (
      <Button type="button" variant="secondary" size="xs" onClick={onRetry}>
        Retry
      </Button>
    ),
  });
}

/** The task was archived or removed elsewhere. */
export function showGone() {
  toast({ title: "This task no longer exists", variant: "destructive" });
}

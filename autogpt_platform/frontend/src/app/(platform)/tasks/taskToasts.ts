import type { TaskUpdateRequest } from "@/app/api/__generated__/models/taskUpdateRequest";
import { toast } from "@/components/molecules/Toast/use-toast";

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

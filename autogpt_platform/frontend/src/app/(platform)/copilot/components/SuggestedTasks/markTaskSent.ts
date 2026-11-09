import {
  getGetBrainDumpSuggestedTasksQueryKey,
  updateBrainDumpSuggestedTasks,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { getQueryClient } from "@/lib/react-query/queryClient";

const ATTEMPTS = 3;

// For a task started outside the list, such as from the trial cancel offer.
// Each write sets a state, so a retry is safe; nobody is watching the row
// here, so one that keeps failing stays quiet and the row stays open.
export async function markTaskSent({
  taskID,
  sessionID,
}: {
  taskID: string;
  sessionID: string;
}) {
  for (let attempt = 0; attempt < ATTEMPTS; attempt += 1) {
    try {
      await updateBrainDumpSuggestedTasks({
        task_ids: [taskID],
        state: "sent",
        session_id: sessionID,
      });
      await getQueryClient().invalidateQueries({
        queryKey: getGetBrainDumpSuggestedTasksQueryKey(),
      });
      return;
    } catch {
      // Try again, up to the limit.
    }
  }
}

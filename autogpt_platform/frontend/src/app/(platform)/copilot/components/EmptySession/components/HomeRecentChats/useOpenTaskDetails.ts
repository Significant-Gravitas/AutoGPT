import type { Task } from "@/app/api/__generated__/models/task";
import { useIsMobile } from "@/app/(platform)/copilot/useIsMobile";
import { useWorkAreaStore } from "@/app/(platform)/copilot/workArea";
import { BOARD_HREF, taskHref } from "@/app/(platform)/tasks/helpers";
import { useRouter } from "next/navigation";

/** Opens a task the way Home's brief does: in the work area beside Home, or
 *  on a phone, which has no room for one, in its chat with the task shown. */
export function useOpenTaskDetails() {
  const router = useRouter();
  const isMobile = useIsMobile();
  const openTab = useWorkAreaStore((s) => s.openTask);

  function openTaskDetails(task: Task) {
    if (isMobile) router.push(taskHref(task) ?? BOARD_HREF);
    else openTab(task.id, task.title);
  }

  return { openTaskDetails };
}

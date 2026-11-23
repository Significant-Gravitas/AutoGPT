import { toast } from "@/components/molecules/Toast/use-toast";
import { BOARD_HREF, TASK_PARAM } from "./helpers";

/** The path that opens a task beside the board on Home: the same `?task=`
 *  the board writes when a task is open. */
export function taskLinkPath(taskID: string) {
  return `${BOARD_HREF}&${TASK_PARAM}=${encodeURIComponent(taskID)}`;
}

/** The full link to a task, to paste anywhere. */
export function taskLinkURL(taskID: string, origin = window.location.origin) {
  return new URL(taskLinkPath(taskID), origin).href;
}

/** Copy a task's link, saying so; where the browser won't allow it, the
 *  toast shows the link to copy by hand. */
export async function copyTaskLink(taskID: string) {
  const url = taskLinkURL(taskID);
  try {
    if (!navigator.clipboard) throw new Error("No clipboard");
    await navigator.clipboard.writeText(url);
    toast({ title: "Link copied" });
  } catch {
    toast({ title: "Copy this link to share the task", description: url });
  }
}

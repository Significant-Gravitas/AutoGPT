import { TASK_PARAM } from "@/app/(platform)/tasks/helpers";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useRef } from "react";
import { useWorkAreaStore } from "../../../../workArea";

const SEPARATOR = ",";

/** Keeps the open task in the address, so a reload or a shared link opens
 *  it again: the task in front of the sidebar is `?task=`, and closing
 *  that task's tab takes it off. The sidebar reads `?task=` on load. */
export function useTaskLink() {
  const tabs = useWorkAreaStore((s) => s.tabs);
  const activeKey = useWorkAreaStore((s) => s.activeKey);
  const [linked, setLinked] = useQueryState(
    TASK_PARAM,
    parseAsString.withOptions({ history: "replace" }),
  );
  const front = tabs.find((tab) => tab.key === activeKey);
  const frontTask = front?.kind === "task" ? front.taskID : null;
  const openTasks = tabs
    .flatMap((tab) => (tab.kind === "task" ? [tab.taskID] : []))
    .join(SEPARATOR);
  // A link read on load is kept until its tab has opened and then closed.
  const wasOpen = useRef<string[]>([]);

  useEffect(() => {
    const open = openTasks ? openTasks.split(SEPARATOR) : [];
    const was = wasOpen.current;
    wasOpen.current = open;
    if (frontTask) {
      if (frontTask !== linked) void setLinked(frontTask);
      return;
    }
    if (linked && was.includes(linked) && !open.includes(linked)) {
      void setLinked(null);
    }
  }, [frontTask, openTasks, linked, setLinked]);
}

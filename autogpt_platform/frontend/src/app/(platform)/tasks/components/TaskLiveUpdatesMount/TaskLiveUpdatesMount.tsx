"use client";

import { isTasksV1Enabled } from "@/app/(platform)/tasks/data/flag";
import { useTaskLiveUpdatesMount } from "./useTaskLiveUpdatesMount";

export function TaskLiveUpdatesMount() {
  if (!isTasksV1Enabled()) return null;
  return <LiveUpdates />;
}

function LiveUpdates() {
  useTaskLiveUpdatesMount();
  return null;
}

"use client";

import { TasksWorkspace } from "./components/TasksWorkspace/TasksWorkspace";

export default function TasksPage() {
  return (
    <TasksWorkspace className="flex h-[calc(100svh-4.5rem)] min-h-0 w-full flex-col px-4 pb-4 md:h-(--inset-height) md:px-6 md:pt-6" />
  );
}

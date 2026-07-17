"use client";

import { ArrowRightIcon, ListChecksIcon } from "@phosphor-icons/react";
import { useCopilotUIStore } from "../../../store";

export function TaskListNotice() {
  const openContextPanelForProgress = useCopilotUIStore(
    (s) => s.openContextPanelForProgress,
  );

  return (
    <button
      type="button"
      onClick={openContextPanelForProgress}
      className="inline-flex w-fit items-center gap-2 rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs text-zinc-600 transition-colors hover:border-zinc-300 hover:bg-zinc-50"
    >
      <ListChecksIcon size={14} className="text-blue-500" />
      <span>Progress shown in the sidebar</span>
      <ArrowRightIcon size={12} className="text-zinc-400" />
    </button>
  );
}

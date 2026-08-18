"use client";
import { useCopilotUIStore } from "../../../store";
import { ArrowRight02Icon, CheckListIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
      <Icon icon={CheckListIcon} size={14} className="text-blue-500" />
      <span>Progress shown in the sidebar</span>
      <Icon icon={ArrowRight02Icon} size={12} className="text-zinc-400" />
    </button>
  );
}

"use client";

import { SidebarSimpleIcon } from "@phosphor-icons/react";
import { useCopilotUIStore } from "../../store";

export function ContextPanelToggle() {
  const isOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);
  const toggleContextPanel = useCopilotUIStore((s) => s.toggleContextPanel);

  if (isOpen) return null;

  return (
    <div className="flex shrink-0 items-start p-3">
      <button
        type="button"
        onClick={toggleContextPanel}
        aria-label="Open workspace panel"
        aria-pressed={false}
        className="rounded-md bg-white/80 p-2 text-zinc-500 shadow-sm backdrop-blur transition-colors hover:bg-white hover:text-zinc-700"
      >
        <SidebarSimpleIcon size={18} />
      </button>
    </div>
  );
}

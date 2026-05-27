"use client";

import { SidebarSimpleIcon } from "@phosphor-icons/react";
import { useCopilotUIStore } from "../../store";

export function ContextPanelToggle() {
  const isOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);
  const toggleContextPanel = useCopilotUIStore((s) => s.toggleContextPanel);

  return (
    <button
      type="button"
      onClick={toggleContextPanel}
      aria-label={isOpen ? "Close workspace panel" : "Open workspace panel"}
      aria-pressed={isOpen}
      className="absolute right-3 top-3 z-20 rounded-md bg-white/80 p-2 text-zinc-500 shadow-sm backdrop-blur transition-colors hover:bg-white hover:text-zinc-700"
    >
      <SidebarSimpleIcon size={18} />
    </button>
  );
}

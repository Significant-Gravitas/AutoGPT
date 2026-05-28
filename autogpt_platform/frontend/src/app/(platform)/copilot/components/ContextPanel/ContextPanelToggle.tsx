"use client";

import { Button } from "@/components/ui/button";
import { SidebarSimpleIcon } from "@phosphor-icons/react";
import { useCopilotUIStore } from "../../store";

export function ContextPanelToggle() {
  const isOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);
  const toggleContextPanel = useCopilotUIStore((s) => s.toggleContextPanel);

  if (isOpen) return null;

  return (
    <div className="flex shrink-0 items-start p-3">
      <Button
        type="button"
        variant="ghost"
        size="icon"
        onClick={toggleContextPanel}
        aria-label="Open workspace panel"
        aria-pressed={false}
      >
        <SidebarSimpleIcon className="!size-5 rotate-180" />
      </Button>
    </div>
  );
}

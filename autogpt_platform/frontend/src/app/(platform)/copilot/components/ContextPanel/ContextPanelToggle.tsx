"use client";

import { Button } from "@/components/ui/button";
import { useCopilotUIStore } from "../../store";
import { Folder01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function ContextPanelToggle() {
  const isOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);
  const hasArtifact = useCopilotUIStore(
    (s) => s.artifactPanel.activeArtifact != null,
  );
  const toggleContextPanel = useCopilotUIStore((s) => s.toggleContextPanel);

  if (isOpen || hasArtifact) return null;

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
        <Icon icon={Folder01Icon} className="!size-5" />
      </Button>
    </div>
  );
}

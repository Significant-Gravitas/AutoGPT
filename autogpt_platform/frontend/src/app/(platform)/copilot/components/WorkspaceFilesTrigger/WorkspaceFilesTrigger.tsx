"use client";

import { Button } from "@/components/ui/button";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Folder01Icon } from "@hugeicons/core-free-icons";
import { useCopilotUIStore } from "../../store";

interface Props {
  className?: string;
}

// Sits next to the sidebar toggle in the new-layout inset header and mirrors
// SidebarTrigger's ghost styling (no border, no shadow).
export function WorkspaceFilesTrigger({ className }: Props) {
  const toggleContextPanel = useCopilotUIStore((s) => s.toggleContextPanel);
  const isContextPanelEnabled = useGetFlag(Flag.ARTIFACTS);

  if (!isContextPanelEnabled) return null;

  return (
    <Button
      variant="ghost"
      size="icon"
      className={className}
      onClick={toggleContextPanel}
      aria-label="Open workspace files"
    >
      <Icon icon={Folder01Icon} className="!size-5" />
    </Button>
  );
}

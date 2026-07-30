"use client";

import { Button } from "@/components/ui/button";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { FolderIcon } from "@phosphor-icons/react";
import { useCopilotUIStore } from "../../store";

interface Props {
  className?: string;
}

// Sits next to the sidebar toggle in the new-layout inset header and mirrors
// SidebarTrigger's ghost styling (no border, no shadow). Replaces the old
// floating "Open workspace files" button from MobileHeader.
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
      <FolderIcon className="!size-5" />
    </Button>
  );
}

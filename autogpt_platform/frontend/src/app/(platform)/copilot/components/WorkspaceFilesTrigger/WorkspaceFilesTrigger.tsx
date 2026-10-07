"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
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

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      className={cn(
        "size-9 rounded-md hover:border-transparent hover:bg-zinc-100",
        className,
      )}
      onClick={toggleContextPanel}
      aria-label="Open workspace files"
      withTooltip={false}
    >
      <Icon icon={Folder01Icon} className="size-5" />
    </Button>
  );
}

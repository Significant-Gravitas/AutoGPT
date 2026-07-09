"use client";

import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ListAiIcon } from "@/components/icons/ListAiIcon";
import { usePathname } from "next/navigation";
import { parseAsString, useQueryState } from "nuqs";
import { useCopilotUIStore } from "../../store";
import { useIsMobile } from "../../useIsMobile";

/**
 * Header affordance to toggle the workspace (files) card open/closed.
 * Self-guards so it can live in the generic inset header — no-ops off the
 * copilot route and hides while a file preview is open.
 */
export function ContextPanelToggle() {
  const pathname = usePathname();
  const isMobile = useIsMobile();
  const isArtifactsEnabled = useGetFlag(Flag.ARTIFACTS);
  const [sessionId] = useQueryState("sessionId", parseAsString);
  const isOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);
  const hasArtifact = useCopilotUIStore(
    (s) => s.artifactPanel.activeArtifact != null,
  );
  const toggleContextPanel = useCopilotUIStore((s) => s.toggleContextPanel);

  if (
    isMobile ||
    pathname !== "/copilot" ||
    !isArtifactsEnabled ||
    !sessionId ||
    hasArtifact
  ) {
    return null;
  }

  return (
    <button
      type="button"
      aria-label={isOpen ? "Close workspace panel" : "Open workspace panel"}
      aria-pressed={isOpen}
      onClick={toggleContextPanel}
      className={cn(
        "flex items-center justify-center rounded-lg p-1.5 transition-colors",
        isOpen
          ? "bg-zinc-100 text-zinc-900"
          : "text-zinc-700 hover:bg-zinc-100 hover:text-zinc-900",
      )}
    >
      <ListAiIcon size={18} />
    </button>
  );
}

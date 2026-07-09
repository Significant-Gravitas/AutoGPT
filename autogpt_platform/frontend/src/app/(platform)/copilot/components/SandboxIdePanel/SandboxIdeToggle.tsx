"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { SidebarSimpleIcon } from "@/components/atoms/AGPTIcon/icons";
import { usePathname } from "next/navigation";
import { parseAsString, useQueryState } from "nuqs";
import { useCopilotUIStore } from "../../store";
import { useIsMobile } from "../../useIsMobile";

/**
 * Header affordance to open the Sandbox IDE panel (new AutoGPT layout only).
 * Only shown while the panel is closed — the open panel has its own close (X).
 * Self-guards so it can live in the generic inset header — no-ops off the
 * copilot route.
 */
export function SandboxIdeToggle() {
  const pathname = usePathname();
  const isMobile = useIsMobile();
  // Child of the new layout: the IDE only shows when both flags are on.
  const isNewLayout = useGetFlag(Flag.AUTOGPT_NEW_LAYOUT);
  const isIdeFlagEnabled = useGetFlag(Flag.AUTOGPT_NEW_LAYOUT_IDE);
  const isEnabled = isNewLayout && isIdeFlagEnabled;
  const [sessionId] = useQueryState("sessionId", parseAsString);
  const isOpen = useCopilotUIStore((s) => s.sandboxIdePanel.isOpen);
  const openSandboxIde = useCopilotUIStore((s) => s.openSandboxIde);

  if (
    isMobile ||
    pathname !== "/copilot" ||
    !isEnabled ||
    !sessionId ||
    isOpen
  ) {
    return null;
  }

  return (
    <button
      type="button"
      aria-label="Open sandbox"
      onClick={openSandboxIde}
      className="flex items-center justify-center rounded-lg p-1.5 text-zinc-700 transition-colors hover:bg-zinc-100 hover:text-zinc-900"
    >
      <SidebarSimpleIcon size={18} className="-scale-x-100" />
    </button>
  );
}

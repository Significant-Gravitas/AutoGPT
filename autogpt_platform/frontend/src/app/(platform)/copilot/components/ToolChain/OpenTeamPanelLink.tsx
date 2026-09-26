"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useCopilotUIStore } from "../../store";

/** Inline link from a delegation card to the Team panel, where the full
 *  roster lives. The chat card stays minimal on purpose. */
export function OpenTeamPanelLink() {
  const isArtifactsEnabled = useGetFlag(Flag.ARTIFACTS);
  const openContextPanelTab = useCopilotUIStore((s) => s.openContextPanelTab);
  if (!isArtifactsEnabled) return null;
  return (
    <button
      type="button"
      onClick={() => openContextPanelTab("team")}
      className="mt-1.5 self-start rounded-sm pl-9 text-xs text-zinc-400 transition-colors hover:text-zinc-700"
    >
      Open Team panel ›
    </button>
  );
}

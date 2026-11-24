"use client";

import { SidebarTrigger } from "@/components/molecules/Sidebar/Sidebar";
import { tourTarget } from "@/lib/tour-targets";

import { useIsChatSession } from "../../useIsChatSession";

// Mobile header buttons float over the content, so they get a pill surface to
// stay readable against whatever scrolls underneath.
const mobileTriggerClassName =
  "rounded-full border border-border-base bg-surface shadow-md hover:bg-surface-muted";

export function InsetHeaderSidebarTrigger() {
  const isChatSession = useIsChatSession();
  // An open chat carries the button in its thread chip (ThreadHeader).
  if (isChatSession) return null;

  return (
    <div className="pointer-events-auto md:hidden" {...tourTarget("menu")}>
      <SidebarTrigger className={mobileTriggerClassName} />
    </div>
  );
}

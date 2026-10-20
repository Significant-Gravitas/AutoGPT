"use client";

import { SidebarTrigger } from "@/components/ui/sidebar";

import { useIsChatSession } from "../../useIsChatSession";

// Mobile header buttons float over the content, so they get a pill surface to
// stay readable against whatever scrolls underneath.
const mobileTriggerClassName =
  "rounded-full border border-[#DADADC] bg-white shadow-md hover:bg-[#F5F5F6]";

export function InsetHeaderSidebarTrigger() {
  const isChatSession = useIsChatSession();
  // An open chat carries the button in its thread chip (ThreadHeader).
  if (isChatSession) return null;

  return (
    <div className="pointer-events-auto md:hidden">
      <SidebarTrigger className={mobileTriggerClassName} />
    </div>
  );
}

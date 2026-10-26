"use client";

import { UserGroupIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useCopilotUIStore } from "../../../../store";
import { LiveDelegationProbes } from "../../../DelegationStatusLine/LiveDelegationProbes";
import { useChatSessionDelegations } from "../../useChatSessionDelegations";
import { countWaiting } from "./helpers";

interface Props {
  sessionId: string | null;
  className?: string;
}

/** Opens the Work tab once the chat has handed anything to an expert, with
 *  a count of the hand-offs waiting on the user: approvals plus questions. */
export function WorkToggle({ sessionId, className }: Props) {
  const { delegations, liveStatuses, reportStatus } =
    useChatSessionDelegations(sessionId);
  const toggleContextPanelTab = useCopilotUIStore(
    (s) => s.toggleContextPanelTab,
  );
  const isActive = useCopilotUIStore(
    (s) =>
      s.artifactPanel.isOpen &&
      s.artifactPanel.activeTab === "work" &&
      s.artifactPanel.activeArtifact == null &&
      !s.artifactPanel.isComputerOpen,
  );

  if (delegations.length === 0) return null;
  const waiting = countWaiting(delegations, liveStatuses);

  return (
    <>
      <LiveDelegationProbes delegations={delegations} onStatus={reportStatus} />
      <Button
        type="button"
        variant="ghost"
        size="icon"
        onClick={() => toggleContextPanelTab("work")}
        aria-label={`${isActive ? "Hide" : "Open"} work${
          waiting > 0 ? ` (${waiting} waiting on you)` : ""
        }`}
        aria-pressed={isActive}
        className={cn(
          className,
          "h-8",
          waiting > 0 ? "w-auto gap-1 px-2" : "w-8",
          isActive && "bg-zinc-100",
        )}
      >
        <Icon
          icon={UserGroupIcon}
          className="!size-4 text-sidebar-foreground/90"
        />
        {waiting > 0 && (
          <span
            data-testid="work-badge"
            className="inline-flex min-w-4 items-center justify-center rounded-full bg-amber-100 px-1.5 text-[11px] font-medium leading-4 text-amber-800"
          >
            {waiting}
          </span>
        )}
      </Button>
    </>
  );
}

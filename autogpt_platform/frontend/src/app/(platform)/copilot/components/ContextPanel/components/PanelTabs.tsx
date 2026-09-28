"use client";

import { LicenseDraftIcon, UserGroupIcon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { type ContextPanelTab, useCopilotUIStore } from "../../../store";
import { WorkBadge } from "./WorkBadge";

const TABS: { tab: ContextPanelTab; label: string; icon: IconSvgElement }[] = [
  { tab: "artifacts", label: "Artifacts", icon: LicenseDraftIcon },
  { tab: "work", label: "Work", icon: UserGroupIcon },
];

interface Props {
  sessionId: string | null;
}

/** The docked panel's two faces: what this chat produced, and what it
 *  handed to experts. */
export function PanelTabs({ sessionId }: Props) {
  const activeTab = useCopilotUIStore((s) => s.artifactPanel.activeTab);
  const openWorkTab = useCopilotUIStore((s) => s.openWorkTab);
  const toggleContextPanelTab = useCopilotUIStore(
    (s) => s.toggleContextPanelTab,
  );

  function select(tab: ContextPanelTab) {
    if (tab === activeTab) return;
    if (tab === "work") openWorkTab();
    else toggleContextPanelTab(tab);
  }

  return (
    <div
      role="tablist"
      className="flex shrink-0 items-center gap-4 border-b border-zinc-200 px-4"
    >
      {TABS.map(({ tab, label, icon }) => {
        const active = tab === activeTab;
        return (
          <button
            key={tab}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => select(tab)}
            className={cn(
              "-mb-px flex h-10 items-center gap-1.5 border-b-2 px-1 text-sm transition-colors",
              active
                ? "border-zinc-900 font-medium text-zinc-900"
                : "border-transparent text-zinc-500 hover:text-zinc-800",
            )}
          >
            <Icon icon={icon} size={16} />
            {label}
            {tab === "work" && <WorkBadge sessionId={sessionId} />}
          </button>
        );
      })}
    </div>
  );
}

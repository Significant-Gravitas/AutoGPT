"use client";

import { TestTube01Icon } from "@hugeicons/core-free-icons";
import type { CSSProperties } from "react";

import { WorkTabs } from "@/app/(platform)/copilot/components/WorkArea/WorkTabs";
import type { EvalCase } from "@/app/api/__generated__/models/evalCase";

import type { useCaseTabs } from "../../useCaseTabs";
import { CaseTab } from "./CaseTab";

interface Props {
  tabs: ReturnType<typeof useCaseTabs>;
  casesById: Map<string, EvalCase>;
}

/** The open cases beside the page, laid out like the chat's work area: tabs
 *  across the head, the case's blocks below, straight on the page. */
export function CaseSidePanel({ tabs, casesById }: Props) {
  if (!tabs.isOpen) return null;
  const active = tabs.active ? casesById.get(tabs.active) : undefined;
  return (
    <aside
      aria-label="Open cases"
      style={{ "--case-panel-width": `${tabs.width}px` } as CSSProperties}
      className="flex h-full w-(--case-panel-width) shrink-0 flex-col py-2.5 pr-2.5 pl-1.5 transition-[width] duration-300 ease-out motion-reduce:transition-none"
    >
      <WorkTabs
        tabs={tabs.tabs.map((id) => ({
          key: id,
          title: id,
          icon: TestTube01Icon,
        }))}
        activeKey={tabs.active}
        onSelect={tabs.select}
        onClose={tabs.close}
        isWide={tabs.isWide}
        onToggleWidth={tabs.toggleWidth}
        onFold={tabs.fold}
      />
      <div
        role="tabpanel"
        className="scrollbar-none min-h-0 flex-1 overflow-y-auto overscroll-contain pt-3 pb-3"
      >
        {active ? <CaseTab key={active.case_id} evalCase={active} /> : null}
      </div>
    </aside>
  );
}

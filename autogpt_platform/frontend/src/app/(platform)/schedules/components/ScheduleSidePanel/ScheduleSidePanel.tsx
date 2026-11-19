"use client";

import { RepeatIcon } from "@hugeicons/core-free-icons";
import type { CSSProperties } from "react";

import { WorkTabs } from "@/app/(platform)/copilot/components/WorkArea/WorkTabs";

import { ScheduleDetail, type Schedule } from "../../affect/agent-schedules";
import { LABELS } from "../../helpers";
import type { useScheduleTabs } from "../../useScheduleTabs";
import type { useScheduleDetail } from "./useScheduleDetail";

interface Props {
  tabs: ReturnType<typeof useScheduleTabs>;
  schedules: Schedule[];
  detail: ReturnType<typeof useScheduleDetail>;
  onPause: (row: string, paused: boolean) => void;
  onRunNow: (row: string) => void;
}

/** The open schedules beside the page, laid out like the chat's work area:
 *  tabs across the head, the block's Schedule panel below, on the page. */
export function ScheduleSidePanel({
  tabs,
  schedules,
  detail,
  onPause,
  onRunNow,
}: Props) {
  if (!tabs.isOpen) return null;
  const names = new Map(schedules.map((s) => [s.id, s.name]));
  return (
    <aside
      aria-label="Open schedules"
      style={{ "--schedule-panel-width": `${tabs.width}px` } as CSSProperties}
      className="flex h-full w-(--schedule-panel-width) shrink-0 flex-col py-2.5 pr-2.5 pl-1.5 transition-[width] duration-300 ease-out motion-reduce:transition-none"
    >
      <WorkTabs
        tabs={tabs.tabs.map((id) => ({
          key: id,
          title: names.get(id) ?? "Schedule",
          icon: RepeatIcon,
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
        <ScheduleDetail
          key={tabs.active ?? "none"}
          schedule={detail.schedule}
          labels={LABELS}
          onPause={onPause}
          onRunNow={onRunNow}
          actions={detail.actions}
        />
      </div>
    </aside>
  );
}

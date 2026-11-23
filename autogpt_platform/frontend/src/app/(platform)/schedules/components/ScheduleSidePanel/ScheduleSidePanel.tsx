"use client";

import { RepeatIcon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { CSSProperties } from "react";
import { createPortal } from "react-dom";

import { useInsetSideSlot } from "@/components/layout/InsetSidePanel/InsetSidePanel";
import { WorkTabs } from "@/app/(platform)/copilot/components/WorkArea/WorkTabs";

import { ScheduleDetail, type Schedule } from "../../affect/agent-schedules";
import { LABELS } from "../../helpers";
import type { useScheduleTabs } from "../../useScheduleTabs";
import type { useScheduleDetail } from "./useScheduleDetail";

const EASE_OUT_QUART: [number, number, number, number] = [0.25, 1, 0.5, 1];
const SLIDE_PX = 24;

interface Props {
  tabs: ReturnType<typeof useScheduleTabs>;
  schedules: Schedule[];
  detail: ReturnType<typeof useScheduleDetail>;
  onPause: (row: string, paused: boolean) => void;
  onRunNow: (row: string) => void;
}

/** The open schedules beside the page, in the same place and look as the
 *  chat's work area: outside the page's card, on the app's own ground with
 *  no fill, tabs across the head and the block's Schedule panel below. */
export function ScheduleSidePanel({
  tabs,
  schedules,
  detail,
  onPause,
  onRunNow,
}: Props) {
  const slot = useInsetSideSlot();
  const reduceMotion = useReducedMotion();
  const speed = reduceMotion ? 0 : 1;
  const names = new Map(schedules.map((s) => [s.id, s.name]));
  // The same open and close as the chat's work area: the panel's width
  // grows from nothing while its content slides in and fades up.
  const panel = (
    <AnimatePresence initial={false}>
      {tabs.isOpen ? (
        <motion.aside
          aria-label="Open schedules"
          initial={{ width: 0 }}
          animate={{ width: tabs.width }}
          exit={{ width: 0 }}
          transition={{ duration: 0.42 * speed, ease: EASE_OUT_QUART }}
          className={
            slot
              ? "sticky top-0 h-svh shrink-0 self-start overflow-hidden select-text"
              : "h-full shrink-0 overflow-hidden"
          }
        >
          <motion.div
            initial={{ opacity: 0, x: SLIDE_PX * speed }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: SLIDE_PX * speed }}
            transition={{ duration: 0.32 * speed, ease: EASE_OUT_QUART }}
            style={
              { "--schedule-panel-width": `${tabs.width}px` } as CSSProperties
            }
            className="flex h-full w-(--schedule-panel-width) flex-col py-2.5 pr-2.5 pl-1.5"
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
          </motion.div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  );
  return slot ? createPortal(panel, slot) : panel;
}

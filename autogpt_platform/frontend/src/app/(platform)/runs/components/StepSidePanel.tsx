"use client";

import { CubeIcon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { CSSProperties } from "react";
import { createPortal } from "react-dom";

import { WorkTabs } from "@/app/(platform)/copilot/components/WorkArea/WorkTabs";
import { useInsetSideSlot } from "@/components/layout/InsetSidePanel/InsetSidePanel";

import type { useStepTabs } from "../useStepTabs";
import { RunStepView, type RunScreenData } from "./RunScreen";

const EASE_OUT_QUART: [number, number, number, number] = [0.25, 1, 0.5, 1];
const SLIDE_PX = 24;

interface Props {
  tabs: ReturnType<typeof useStepTabs>;
  data: RunScreenData;
}

/** The open block runs beside the run, in the same place and look as the
 *  chat's work area: outside the page's card, on the app's own ground with
 *  no fill, tabs across the head and the block's step panel below. */
export function StepSidePanel({ tabs, data }: Props) {
  const slot = useInsetSideSlot();
  const reduceMotion = useReducedMotion();
  const speed = reduceMotion ? 0 : 1;
  const names = new Map(data.trace.steps.map((step) => [step.id, step.name]));
  const panel = (
    <AnimatePresence initial={false}>
      {tabs.isOpen ? (
        <motion.aside
          aria-label="Open block runs"
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
            style={{ "--step-panel-width": `${tabs.width}px` } as CSSProperties}
            className="flex h-full w-(--step-panel-width) flex-col py-2.5 pr-2.5 pl-1.5"
          >
            <WorkTabs
              tabs={tabs.tabs.map((id) => ({
                key: id,
                title: names.get(id) ?? "Block",
                icon: CubeIcon,
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
              <RunStepView {...data} stepID={tabs.active} embed />
            </div>
          </motion.div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  );
  return slot ? createPortal(panel, slot) : panel;
}

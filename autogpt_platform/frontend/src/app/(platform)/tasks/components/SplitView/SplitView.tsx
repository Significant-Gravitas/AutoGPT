"use client";

import { Button } from "@/components/atoms/Button/Button";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion } from "framer-motion";
import type { ReactNode } from "react";
import { PanelCard } from "./components/PanelCard";
import { ResizeHandle } from "./components/ResizeHandle";
import { useSplitView } from "./useSplitView";

const PANEL_ID = "tasks-detail-panel";

interface Props {
  selectedKey: string | null;
  list: ReactNode;
  panel: ReactNode;
  onBack(): void;
}

export function SplitView({ selectedKey, list, panel, onBack }: Props) {
  const split = useSplitView(selectedKey);
  const isOpen = !!selectedKey;

  if (!split.isDesktop && isOpen) {
    return (
      <div className="flex h-full min-h-0 flex-col gap-2">
        <div>
          <Button
            variant="ghost"
            size="xs"
            leadingIcon={ArrowLeft01Icon}
            onClick={onBack}
          >
            Back to tasks
          </Button>
        </div>
        <div id={PANEL_ID} className="min-h-0 flex-1">
          <PanelCard>{panel}</PanelCard>
        </div>
      </div>
    );
  }

  const showPanel = isOpen && !split.isCollapsed;

  return (
    <div ref={split.attachRoot} className="flex h-full min-h-0 w-full">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">{list}</div>
      <AnimatePresence initial={false}>
        {showPanel ? (
          <motion.aside
            key="panel"
            id={PANEL_ID}
            aria-label="Task details"
            className="flex h-full min-h-0 shrink-0 justify-end overflow-hidden"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: split.width, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={split.transition}
          >
            <div
              className="flex h-full shrink-0"
              // oxlint-disable-next-line design-system/no-inline-styles -- the width the user dragged the panel to
              style={{ width: split.width }}
            >
              <ResizeHandle
                panelID={PANEL_ID}
                width={split.width}
                maxWidth={split.maxWidth}
                onCollapse={split.collapse}
                handlers={split.handle}
              />
              <div className="min-h-0 min-w-0 flex-1">
                <PanelCard>{panel}</PanelCard>
              </div>
            </div>
          </motion.aside>
        ) : null}
      </AnimatePresence>
      {isOpen && split.isCollapsed ? (
        <div className="flex h-full shrink-0 pl-3">
          <div className="rounded-card bg-surface p-1 shadow-card">
            <Button
              variant="icon"
              size="icon-2xs"
              aria-label="Show task details"
              aria-expanded={false}
              aria-controls={PANEL_ID}
              leadingIcon={ArrowLeft01Icon}
              onClick={split.expand}
            />
          </div>
        </div>
      ) : null}
    </div>
  );
}

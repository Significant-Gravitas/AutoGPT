"use client";

import type { TaskArtifact } from "@/app/api/__generated__/models/taskArtifact";
import { useInsetSideSlot } from "@/components/layout/InsetSidePanel/InsetSidePanel";
import { ComputerIcon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { MIN_ARTIFACT_PANEL_WIDTH } from "../../store";
import {
  COMPUTER_TAB_KEY,
  CONTEXT_TAB_KEY,
  START_TAB_KEY,
} from "../../workArea";
import { ComputerPanelContent } from "../ArtifactPanel/components/ComputerPanelContent";
import { ContextPanelBody } from "../ContextPanel/ContextPanel";
import { PanelResizeHandle } from "../PanelResizeHandle";
import { ArtifactTab } from "./ArtifactTab";
import { BriefTab } from "./BriefTab";
import { FilesTab } from "./FilesTab";
import { OverviewTab } from "./OverviewTab/OverviewTab";
import { TaskTab } from "./TaskTab";
import { useWorkArea } from "./useWorkArea";
import { TAB_ICONS, WorkTabs, type TabView } from "./WorkTabs";

const EASE_OUT_QUART: [number, number, number, number] = [0.25, 1, 0.5, 1];
const SLIDE_PX = 16;

interface Props {
  /** The open chat; none on Home, where the sidebar holds only what the
   *  board opened. */
  sessionId: string | null;
}

/** The chat's one right sidebar: it sits straight on the page beside the
 *  conversation, with no fill of its own, with tabs for what was opened
 *  from it. It slides
 *  in from the right as it fades up; closing the last tab folds it away. */
export function WorkArea({ sessionId }: Props) {
  const area = useWorkArea(sessionId);
  const [isResizing, setIsResizing] = useState(false);
  const reduceMotion = useReducedMotion();
  // The panel sits in the page's side slot, beside the main panel; a page
  // frame without one keeps it in place, next to the chat.
  const slot = useInsetSideSlot();
  const slide = reduceMotion ? 0 : SLIDE_PX;
  const speed = reduceMotion || isResizing ? 0 : 1;

  const tabViews: TabView[] = [
    ...(area.hasContext
      ? [
          {
            key: CONTEXT_TAB_KEY,
            title: area.contextTitle,
            icon: TAB_ICONS.context,
          },
        ]
      : []),
    ...area.tabs.map((tab) => ({
      key: tab.key,
      title:
        tab.kind === "task" || tab.kind === "brief"
          ? tab.title
          : tab.kind === "artifact"
            ? tab.artifact.title
            : "This chat",
      icon: TAB_ICONS[tab.kind],
    })),
    ...(area.hasComputer
      ? [{ key: COMPUTER_TAB_KEY, title: "Computer", icon: ComputerIcon }]
      : []),
  ];
  const active = area.tabs.find((tab) => tab.key === area.activeKey) ?? null;
  // The viewers scroll themselves; the lists scroll with the panel.
  const ownsScroll =
    active?.kind === "artifact" || area.activeKey === COMPUTER_TAB_KEY;

  function openTaskArtifact(artifact: TaskArtifact) {
    if (!artifact.workspace_file_id) return;
    area.openArtifact({
      id: artifact.workspace_file_id,
      title: artifact.title,
      mimeType: null,
      sourceUrl: `/api/proxy/api/workspace/files/${artifact.workspace_file_id}/download`,
      origin: "agent",
    });
  }

  const panel = (
    <AnimatePresence initial={false}>
      {area.isOpen ? (
        <motion.aside
          data-work-area
          aria-label="Work area"
          initial={{ width: 0 }}
          animate={{ width: area.width }}
          exit={{ width: 0 }}
          transition={{ duration: 0.42 * speed, ease: EASE_OUT_QUART }}
          className="sticky top-0 h-svh shrink-0 self-start select-text"
        >
          <PanelResizeHandle
            panelSelector="[data-work-area]"
            onWidthChange={area.setWidth}
            onResizingChange={setIsResizing}
            minWidth={MIN_ARTIFACT_PANEL_WIDTH}
          />
          <div className="h-full overflow-hidden">
            <motion.div
              initial={{ opacity: 0, x: slide }}
              animate={{
                opacity: 1,
                x: 0,
                transition: {
                  opacity: { duration: 0.26 * speed, ease: EASE_OUT_QUART },
                  x: { duration: 0.42 * speed, ease: EASE_OUT_QUART },
                },
              }}
              exit={{
                opacity: 0,
                x: slide,
                transition: {
                  opacity: { duration: 0.2 * speed, ease: EASE_OUT_QUART },
                  x: { duration: 0.42 * speed, ease: EASE_OUT_QUART },
                },
              }}
              style={
                { "--work-area-width": `${area.width}px` } as CSSProperties
              }
              className="flex h-full w-(--work-area-width) flex-col py-2.5 pr-2.5 pl-1.5"
            >
              <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
                <WorkTabs
                  tabs={tabViews}
                  activeKey={area.activeKey}
                  onSelect={area.select}
                  onClose={area.close}
                  onNew={sessionId ? area.openStart : undefined}
                  isWide={area.isWide}
                  onToggleWidth={area.toggleWidth}
                />
                <div
                  role="tabpanel"
                  className={
                    ownsScroll
                      ? "mt-3 flex min-h-0 flex-1 flex-col overflow-hidden"
                      : "scrollbar-none min-h-0 flex-1 overflow-y-auto overscroll-contain pt-3 pb-3"
                  }
                >
                  {sessionId && area.activeKey === CONTEXT_TAB_KEY ? (
                    area.contextTab === "files" ? (
                      <FilesTab sessionId={sessionId} />
                    ) : (
                      <ContextPanelBody sessionId={sessionId} />
                    )
                  ) : sessionId && area.activeKey === COMPUTER_TAB_KEY ? (
                    <ComputerPanelContent sessionId={sessionId} />
                  ) : sessionId && area.activeKey === START_TAB_KEY ? (
                    <OverviewTab
                      sessionId={sessionId}
                      hasExpert={area.hasExpert}
                      onOpen={area.openFromOverview}
                      onOpenTask={area.openTask}
                    />
                  ) : active?.kind === "task" ? (
                    <TaskTab
                      key={active.taskID}
                      taskID={active.taskID}
                      sessionID={sessionId}
                      experts={area.experts}
                      onOpenArtifact={openTaskArtifact}
                    />
                  ) : active?.kind === "brief" ? (
                    <BriefTab />
                  ) : active?.kind === "artifact" ? (
                    <ArtifactTab />
                  ) : null}
                </div>
              </div>
            </motion.div>
          </div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  );

  return slot ? createPortal(panel, slot) : panel;
}

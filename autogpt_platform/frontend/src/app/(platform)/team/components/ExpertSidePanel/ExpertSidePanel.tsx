"use client";

import { Button } from "@/components/atoms/Button/Button";
import { FullscreenDialog } from "@/components/molecules/FullscreenDialog/FullscreenDialog";
import { Text } from "@/components/atoms/Text/Text";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ReactNode, useEffect, useState } from "react";
import { PanelResizeHandle } from "@/app/(platform)/copilot/components/PanelResizeHandle";
import { useIsMobile } from "@/app/(platform)/copilot/useIsMobile";
import { IdentityAvatar, PanelIdentity } from "./IdentityAvatar";
import { usePanelSidebarCollapse } from "./usePanelSidebarCollapse";

const DEFAULT_PANEL_WIDTH = 520;
const MIN_PANEL_WIDTH = 320;
const MAX_PANEL_VIEWPORT_RATIO = 0.4;
const PANEL_EASE: [number, number, number, number] = [0.32, 0.72, 0, 1];
const PANEL_DURATION = 0.3;

interface Props {
  identity: PanelIdentity | null;
  title: string;
  ariaLabel?: string;
  panelId: string;
  closeLabel: string;
  headerActions?: ReactNode;
  showIdentity?: boolean;
  defaultWidth?: number;
  onClose: () => void;
  children: ReactNode;
}

export function ExpertSidePanel({
  identity,
  title,
  ariaLabel = title,
  panelId,
  closeLabel,
  headerActions,
  showIdentity = true,
  defaultWidth = DEFAULT_PANEL_WIDTH,
  onClose,
  children,
}: Props) {
  const isMobile = useIsMobile();
  const shouldReduceMotion = useReducedMotion();
  const [width, setWidth] = useState(defaultWidth);
  const [isResizing, setIsResizing] = useState(false);
  const maxWidth = useMaxPanelWidth();
  const renderedWidth = Math.min(width, maxWidth);
  usePanelSidebarCollapse(identity !== null);

  if (isMobile) {
    if (!identity) return null;
    return (
      <FullscreenDialog title={ariaLabel} onClose={onClose}>
        <PanelHeader
          identity={identity}
          title={title}
          closeLabel={closeLabel}
          actions={headerActions}
          showIdentity={showIdentity}
          onClose={onClose}
        />
        {children}
      </FullscreenDialog>
    );
  }

  const transition =
    shouldReduceMotion || isResizing
      ? { duration: 0 }
      : { duration: PANEL_DURATION, ease: PANEL_EASE };
  const panelSelector = `[data-expert-panel="${panelId}"]`;

  return (
    <AnimatePresence>
      {identity ? (
        <motion.aside
          data-expert-panel={panelId}
          aria-label={ariaLabel}
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: renderedWidth, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={transition}
          className="sticky top-0 h-svh shrink-0 self-start border-l border-l-[#80808017] bg-sidebar"
        >
          <PanelResizeHandle
            panelSelector={panelSelector}
            onWidthChange={setWidth}
            onResizingChange={setIsResizing}
            minWidth={MIN_PANEL_WIDTH}
            maxWidth={maxWidth}
          />
          <div className="h-full overflow-hidden">
            <div
              style={{ width: renderedWidth }}
              className="flex h-full min-h-0 flex-col"
            >
              <PanelHeader
                identity={identity}
                title={title}
                closeLabel={closeLabel}
                actions={headerActions}
                showIdentity={showIdentity}
                onClose={onClose}
              />
              {children}
            </div>
          </div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  );
}

function useMaxPanelWidth() {
  const [maxWidth, setMaxWidth] = useState(DEFAULT_PANEL_WIDTH);

  useEffect(() => {
    function update() {
      setMaxWidth(
        Math.max(
          MIN_PANEL_WIDTH,
          Math.round(window.innerWidth * MAX_PANEL_VIEWPORT_RATIO),
        ),
      );
    }
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  return maxWidth;
}

interface HeaderProps {
  identity: PanelIdentity;
  title: string;
  closeLabel: string;
  actions?: ReactNode;
  showIdentity: boolean;
  onClose: () => void;
}

function PanelHeader({
  identity,
  title,
  closeLabel,
  actions,
  showIdentity,
  onClose,
}: HeaderProps) {
  return (
    <div className="flex h-[53px] shrink-0 items-center gap-2 border-b border-b-[#80808017] px-3">
      {showIdentity ? (
        <>
          <IdentityAvatar
            identity={identity}
            className="h-7 w-7"
            imageSize={56}
          />
          <Text
            variant="body-medium"
            as="h2"
            tone="primary"
            className="min-w-0 flex-1 truncate"
          >
            {title}
          </Text>
        </>
      ) : (
        <span className="flex-1" />
      )}
      {actions}
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        leadingIcon={Cancel01Icon}
        aria-label={closeLabel}
        onClick={onClose}
      />
    </div>
  );
}

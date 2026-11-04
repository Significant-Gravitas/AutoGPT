import { isLargeScreen, useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { isKey } from "@/lib/keyboard";
import { useReducedMotion } from "framer-motion";
import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from "react";
import {
  clampPanelWidth,
  maxPanelWidth,
  PANEL_DEFAULT_WIDTH,
  PANEL_GLIDE,
  PANEL_KEY_STEP,
  readStoredPanelWidth,
  storePanelWidth,
} from "./helpers";

interface Drag {
  startX: number;
  startWidth: number;
}

export function useSplitView(selectedKey: string | null) {
  const isDesktop = isLargeScreen(useBreakpoint());
  const reduceMotion = useReducedMotion();
  const [root, setRoot] = useState<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [storedWidth, setStoredWidth] = useState(PANEL_DEFAULT_WIDTH);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [lastKey, setLastKey] = useState(selectedKey);
  const drag = useRef<Drag | null>(null);

  if (selectedKey !== lastKey) {
    setLastKey(selectedKey);
    setIsCollapsed(false);
  }

  useEffect(() => {
    setStoredWidth(readStoredPanelWidth());
  }, []);

  useEffect(() => {
    if (!root) return;
    const observer = new ResizeObserver(([entry]) => {
      setContainerWidth(entry.contentRect.width);
    });
    observer.observe(root);
    return () => observer.disconnect();
  }, [root]);

  const maxWidth = maxPanelWidth(containerWidth);
  const width = clampPanelWidth(storedWidth, maxWidth);

  function commit(next: number) {
    const clamped = clampPanelWidth(next, maxWidth);
    setStoredWidth(clamped);
    return clamped;
  }

  function widthFromPointer(e: PointerEvent<HTMLElement>) {
    const start = drag.current;
    if (!start) return width;
    return start.startWidth + (start.startX - e.clientX);
  }

  function handlePointerDown(e: PointerEvent<HTMLElement>) {
    if (e.button !== 0) return;
    e.preventDefault();
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // Older test DOMs don't support pointer capture.
    }
    drag.current = { startX: e.clientX, startWidth: width };
    setIsDragging(true);
  }

  function handlePointerMove(e: PointerEvent<HTMLElement>) {
    if (!drag.current) return;
    commit(widthFromPointer(e));
  }

  function handlePointerUp(e: PointerEvent<HTMLElement>) {
    if (!drag.current) return;
    storePanelWidth(commit(widthFromPointer(e)));
    drag.current = null;
    setIsDragging(false);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLElement>) {
    const step = isKey(e, "ArrowLeft")
      ? PANEL_KEY_STEP
      : isKey(e, "ArrowRight")
        ? -PANEL_KEY_STEP
        : 0;
    if (!step) return;
    e.preventDefault();
    storePanelWidth(commit(width + step));
  }

  return {
    isDesktop,
    attachRoot: setRoot,
    width,
    maxWidth: Number.isFinite(maxWidth) ? maxWidth : undefined,
    isCollapsed,
    collapse: () => setIsCollapsed(true),
    expand: () => setIsCollapsed(false),
    transition: reduceMotion || isDragging ? { duration: 0 } : PANEL_GLIDE,
    handle: {
      onPointerDown: handlePointerDown,
      onPointerMove: handlePointerMove,
      onPointerUp: handlePointerUp,
      onPointerCancel: handlePointerUp,
      onKeyDown: handleKeyDown,
    },
  };
}

"use client";

import { useVirtualizer, type Virtualizer } from "@tanstack/react-virtual";
import {
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
  type MutableRefObject,
} from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import {
  estimateRowHeight,
  layoutSegments,
  listPadding,
  mergeRenderedIndexes,
  OVERSCAN_ROWS,
  pinnedTailStart,
  type TranscriptRow,
} from "./helpers";

const ROW_GAP_PX = 24;
const COMPACT_ROW_GAP_PX = 16;

export type JumpToMessage = (messageID: string) => void;

interface Args {
  rows: TranscriptRow[];
  lastUserMessageID: string | null;
  compact: boolean;
  jumpRef?: MutableRefObject<JumpToMessage | null>;
}

/**
 * Windows the transcript to the rows on screen plus a few either side, laid
 * out in normal flow inside StickToBottom's scroll element. The list opts out
 * of the browser's scroll anchoring, which switches itself off in every frame
 * the list's padding changes; instead this hook keeps the view still whenever
 * a row above the viewport changes height.
 */
export function useVirtualTranscript({
  rows,
  lastUserMessageID,
  compact,
  jumpRef,
}: Args) {
  const { scrollRef, state: stickState } = useStickToBottomContext();
  const listRef = useRef<HTMLDivElement>(null);
  const [scrollMargin, setScrollMargin] = useState(0);
  // Null until the first commit; false where the scroll element has no
  // height to window against (jsdom, a hidden panel), which renders every row.
  const [hasViewport, setHasViewport] = useState<boolean | null>(null);
  const rowsRef = useRef(rows);
  rowsRef.current = rows;
  const enteringRef = useRef(false);
  const prependingRef = useRef(false);
  const hasViewportRef = useRef(false);
  hasViewportRef.current = hasViewport === true;
  const firstRowIDRef = useRef<string | undefined>(rows[0]?.message.id);
  const seenIDsRef = useRef(new Set<string>());
  const quietByIDRef = useRef(new Map<string, boolean>());
  const gap = compact ? COMPACT_ROW_GAP_PX : ROW_GAP_PX;

  const getItemKey = useCallback(
    (index: number) => rows[index].message.id,
    [rows],
  );

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (index) => estimateRowHeight(rowsRef.current[index].message),
    getItemKey,
    overscan: OVERSCAN_ROWS,
    gap,
    scrollMargin,
    // A thread opens pinned to its end, so the first pass windows the newest
    // rows instead of mounting the oldest ones only to drop them a frame on.
    initialOffset: Number.MAX_SAFE_INTEGER,
    initialRect: {
      width: 0,
      height: typeof window === "undefined" ? 0 : window.innerHeight,
    },
  });

  virtualizer.shouldAdjustScrollPositionOnItemSizeChange = (
    item,
    delta,
    instance,
  ) => {
    // LoadMoreSentinel restores the position after a prepend, and
    // StickToBottom holds a pinned thread at its end.
    if (
      hasViewportRef.current &&
      !prependingRef.current &&
      !stickState?.isAtBottom
    ) {
      keepRowsBelowStill({
        instance,
        key: item.key,
        delta,
        isEntering: enteringRef.current,
      });
    }
    return false;
  };

  const [measureRow] = useState(() => (node: HTMLDivElement | null) => {
    if (!node) {
      virtualizer.measureElement(null);
      return;
    }
    enteringRef.current =
      virtualizer.elementsCache.get(node.dataset.key ?? "") !== node;
    virtualizer.measureElement(node);
    enteringRef.current = false;
  });

  const firstRowID = rows[0]?.message.id;
  if (firstRowID !== firstRowIDRef.current) {
    const previousFirstID = firstRowIDRef.current;
    firstRowIDRef.current = firstRowID;
    prependingRef.current = shiftOffsetForPrepend({
      virtualizer,
      rows,
      previousFirstID,
      scrollMargin,
    });
  }

  const windowItems = virtualizer.getVirtualItems();
  const measurements = virtualizer.measurementsCache;
  const isWindowed = (hasViewport ?? true) && windowItems.length > 0;
  const indexes = isWindowed
    ? mergeRenderedIndexes(
        windowItems.map((item) => item.index),
        pinnedTailStart(rows, lastUserMessageID),
        rows.length,
      )
    : rows.map((_, index) => index);
  const segments = isWindowed
    ? layoutSegments({ indexes, measurements, gap })
    : indexes.map((index) => ({ kind: "row" as const, index }));
  const padding = isWindowed
    ? listPadding({
        indexes,
        measurements,
        scrollMargin,
        totalSize: virtualizer.getTotalSize(),
      })
    : { top: 0, bottom: 0 };

  const quietByID = quietByIDRef.current;
  for (const index of indexes) {
    const id = rows[index].message.id;
    if (!quietByID.has(id)) quietByID.set(id, seenIDsRef.current.has(id));
  }

  useLayoutEffect(() => {
    prependingRef.current = false;
    const scroller = scrollRef.current;
    const ready = scroller instanceof HTMLElement && scroller.clientHeight > 0;
    if (ready !== hasViewport) setHasViewport(ready);
    if (hasViewport !== null) {
      rememberMountedRows({
        rows,
        indexes,
        seenIDs: seenIDsRef.current,
        quietByID: quietByIDRef.current,
      });
    }
    const list = listRef.current;
    if (!ready || !scroller || !list) return;
    const margin = Math.round(
      list.getBoundingClientRect().top -
        scroller.getBoundingClientRect().top +
        scroller.scrollTop,
    );
    if (margin !== scrollMargin) setScrollMargin(margin);
  });

  useLayoutEffect(() => {
    if (!jumpRef) return;
    jumpRef.current = (messageID) => {
      const index = rowsRef.current.findIndex(
        (row) => row.message.id === messageID,
      );
      if (index >= 0) virtualizer.scrollToIndex(index, { align: "center" });
    };
    return () => {
      jumpRef.current = null;
    };
  }, [jumpRef, virtualizer]);

  return {
    listRef,
    measureRow,
    segments,
    padding,
    isWindowed,
    isQuiet: (id: string) => quietByID.get(id) ?? false,
  };
}

/**
 * A row above the viewport changed height: a row that scrolled into the
 * window replaced padding sized from a guess, or a mounted row grew or shrank
 * (a code block, an image, a tool row). Moving the scroll position by the
 * same amount keeps what the user is reading in place. A row only partly
 * above the viewport is left alone, so a reply streaming below the fold
 * never drags the view.
 */
function keepRowsBelowStill({
  instance,
  key,
  delta,
  isEntering,
}: {
  instance: Virtualizer<HTMLElement, Element>;
  key: string | number | bigint;
  delta: number;
  isEntering: boolean;
}) {
  const scroller = instance.scrollElement;
  const node = instance.elementsCache.get(key);
  if (!(scroller instanceof HTMLElement) || !node) return;
  const viewportTop = scroller.getBoundingClientRect().top;
  const rect = node.getBoundingClientRect();
  // An entering row was padding a moment ago, so it counts from its top; a
  // mounted row counts if it sat wholly above before the change.
  const isAbove = isEntering
    ? rect.top < viewportTop
    : rect.bottom - delta <= viewportTop;
  if (isAbove) scroller.scrollTop += delta;
}

/**
 * Older history landed above the rows on screen. `LoadMoreSentinel` moves the
 * scroll position down by the height that arrived; moving the window's
 * offset by the same amount here keeps the rows the user is reading mounted
 * through that commit, instead of windowing the new rows at the old offset.
 * Returns whether a prepend happened, so the commit leaves scrolling alone.
 */
function shiftOffsetForPrepend({
  virtualizer,
  rows,
  previousFirstID,
  scrollMargin,
}: {
  virtualizer: Virtualizer<HTMLElement, Element>;
  rows: TranscriptRow[];
  previousFirstID: string | undefined;
  scrollMargin: number;
}): boolean {
  if (!previousFirstID || virtualizer.scrollOffset === null) return false;
  const previousIndex = rows.findIndex(
    (row) => row.message.id === previousFirstID,
  );
  if (previousIndex <= 0) return false;
  virtualizer.getTotalSize();
  const previousStart = virtualizer.measurementsCache[previousIndex]?.start;
  if (previousStart === undefined) return false;
  virtualizer.scrollOffset += previousStart - scrollMargin;
  return true;
}

/** Rows mounted from here on, other than brand-new messages, skip the
 *  entrance animation: they were already in the thread and are only
 *  scrolling back into the window. */
function rememberMountedRows({
  rows,
  indexes,
  seenIDs,
  quietByID,
}: {
  rows: TranscriptRow[];
  indexes: number[];
  seenIDs: Set<string>;
  quietByID: Map<string, boolean>;
}) {
  const mounted = new Set(indexes.map((index) => rows[index]?.message.id));
  for (const id of Array.from(quietByID.keys())) {
    if (!mounted.has(id)) quietByID.delete(id);
  }
  for (const row of rows) seenIDs.add(row.message.id);
}

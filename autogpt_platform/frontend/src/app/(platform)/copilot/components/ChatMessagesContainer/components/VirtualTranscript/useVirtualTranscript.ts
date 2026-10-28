"use client";

import {
  observeElementOffset,
  useVirtualizer,
  type Virtualizer,
} from "@tanstack/react-virtual";
import {
  useCallback,
  useLayoutEffect,
  useReducer,
  useRef,
  useState,
  type MutableRefObject,
} from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import {
  estimateRowHeight,
  OVERSCAN_ROWS,
  windowLayout,
  type TranscriptRow,
} from "./helpers";
import {
  keepReadingRowAfterPrepend,
  keepRowsBelowStill,
  listOffset,
  shiftOffsetForPrepend,
  type Prepend,
} from "./scrollAnchoring";
import { useQuietRows } from "./useQuietRows";

const ROW_GAP_PX = 24;
const COMPACT_ROW_GAP_PX = 16;
const MAX_VIEWPORT_RECHECKS = 3;

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
  const [, recheckViewport] = useReducer((count: number) => count + 1, 0);
  const viewportRechecksRef = useRef(0);
  const rowsRef = useRef(rows);
  rowsRef.current = rows;
  const enteringRef = useRef(false);
  const prependRef = useRef<Prepend | null>(null);
  const hasViewportRef = useRef(false);
  hasViewportRef.current = hasViewport === true;
  const firstRowIDRef = useRef<string | undefined>(rows[0]?.message.id);
  const gap = compact ? COMPACT_ROW_GAP_PX : ROW_GAP_PX;

  // Kept as a memo: the compiler skips this hook (TanStack Virtual's
  // `useVirtualizer` is an incompatible library), and the virtualizer redoes
  // every row's measurements whenever `getItemKey` changes identity.
  const getItemKey = useCallback(
    (index: number) => rows[index].message.id,
    [rows],
  );

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (index) => estimateRowHeight(rowsRef.current[index].message),
    getItemKey,
    observeElementOffset: observeOffsetUntilUnsubscribed,
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
      !prependRef.current &&
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
    prependRef.current = shiftOffsetForPrepend({
      virtualizer,
      rows,
      previousFirstID,
      scrollMargin,
    });
  }

  const windowItems = virtualizer.getVirtualItems();
  const measurements = virtualizer.measurementsCache;
  const isWindowed = (hasViewport ?? true) && windowItems.length > 0;
  const { indexes, segments, padding } = windowLayout({
    rows,
    windowIndexes: windowItems.map((item) => item.index),
    isWindowed,
    lastUserMessageID,
    measurements,
    gap,
    scrollMargin,
    totalSize: virtualizer.getTotalSize(),
  });

  const isQuiet = useQuietRows({
    rows,
    indexes,
    isTracking: hasViewport !== null,
  });

  useLayoutEffect(() => {
    const prepend = prependRef.current;
    prependRef.current = null;
    const scroller = scrollRef.current;
    // Mounted together with StickToBottom (a session switch), this runs
    // before React attaches the scroll element's ref (twice under StrictMode).
    // Look again on another layout pass rather than falling back to rendering
    // every row; give up after a few, as with no scroll element at all.
    if (!scroller && viewportRechecksRef.current < MAX_VIEWPORT_RECHECKS) {
      viewportRechecksRef.current += 1;
      recheckViewport();
      return;
    }
    const ready = scroller instanceof HTMLElement && scroller.clientHeight > 0;
    if (prepend && ready) {
      keepReadingRowAfterPrepend({ virtualizer, scroller, prepend });
    }
    if (ready !== hasViewport) setHasViewport(ready);
    const list = listRef.current;
    if (!ready || !scroller || !list) return;
    const margin = listOffset(scroller, list);
    if (margin !== scrollMargin) setScrollMargin(margin);
  });

  useJumpToMessage({ jumpRef, rowsRef, virtualizer });

  return {
    listRef,
    measureRow,
    segments,
    padding,
    isWindowed,
    isQuiet,
  };
}

/** Hands the minimap a way to reach rows the window has unmounted. */
function useJumpToMessage({
  jumpRef,
  rowsRef,
  virtualizer,
}: {
  jumpRef?: MutableRefObject<JumpToMessage | null>;
  rowsRef: MutableRefObject<TranscriptRow[]>;
  virtualizer: ReturnType<typeof useVirtualizer<HTMLElement, Element>>;
}) {
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
  }, [jumpRef, rowsRef, virtualizer]);
}

// virtual-core 3.13 leaves its 150 ms scroll-end timer running on unsubscribe,
// so it updates an unmounted list; 3.17.8 (react-virtual 3.14.10) cancels it.
function observeOffsetUntilUnsubscribed(
  instance: Virtualizer<HTMLElement, Element>,
  cb: (offset: number, isScrolling: boolean) => void,
) {
  let isSubscribed = true;
  const unsubscribe = observeElementOffset(instance, (offset, isScrolling) => {
    if (isSubscribed) cb(offset, isScrolling);
  });
  return () => {
    isSubscribed = false;
    unsubscribe?.();
  };
}

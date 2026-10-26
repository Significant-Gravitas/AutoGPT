import type { Virtualizer } from "@tanstack/react-virtual";
import type { TranscriptRow } from "./helpers";

/** Where the first row sat before older history landed above it. */
export interface Prepend {
  firstRowID: string;
  scrollTop: number;
  firstRowOffset: number;
}

/**
 * A row above the viewport changed height: a row that scrolled into the
 * window replaced padding sized from a guess, or a mounted row grew or shrank
 * (a code block, an image, a tool row). Moving the scroll position by the
 * same amount keeps what the user is reading in place. A row only partly
 * above the viewport is left alone, so a reply streaming below the fold
 * never drags the view.
 */
export function keepRowsBelowStill({
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
 * Older history landed above the rows on screen. Moving the window's offset
 * by the height that arrived keeps the rows the user is reading mounted
 * through the commit, instead of windowing the new rows at the old offset.
 * Returns where the old first row sat, for `keepReadingRowAfterPrepend`.
 */
export function shiftOffsetForPrepend({
  virtualizer,
  rows,
  previousFirstID,
  scrollMargin,
}: {
  virtualizer: Virtualizer<HTMLElement, Element>;
  rows: TranscriptRow[];
  previousFirstID: string | undefined;
  scrollMargin: number;
}): Prepend | null {
  const scrollTop = virtualizer.scrollOffset;
  if (!previousFirstID || scrollTop === null) return null;
  if (scrollTop === Number.MAX_SAFE_INTEGER) return null;
  const previousIndex = rows.findIndex(
    (row) => row.message.id === previousFirstID,
  );
  if (previousIndex <= 0) return null;
  virtualizer.getTotalSize();
  const previousStart = virtualizer.measurementsCache[previousIndex]?.start;
  if (previousStart === undefined) return null;
  virtualizer.scrollOffset = scrollTop + previousStart - scrollMargin;
  return {
    firstRowID: previousFirstID,
    scrollTop,
    firstRowOffset: scrollMargin,
  };
}

/**
 * Puts the row that used to be first back where it was on screen, measured
 * from the DOM after the commit. This runs after `LoadMoreSentinel`'s own
 * restore, which moves by the change in total height: that is only the
 * height that arrived above when nothing below changed, and a page that also
 * merges rows further down would leave the reader somewhere else.
 */
export function keepReadingRowAfterPrepend({
  virtualizer,
  scroller,
  prepend,
}: {
  virtualizer: Virtualizer<HTMLElement, Element>;
  scroller: HTMLElement;
  prepend: Prepend;
}) {
  const node = virtualizer.elementsCache.get(prepend.firstRowID);
  if (!node?.isConnected) return;
  const rowOffset =
    node.getBoundingClientRect().top -
    scroller.getBoundingClientRect().top +
    scroller.scrollTop;
  scroller.scrollTop = prepend.scrollTop + rowOffset - prepend.firstRowOffset;
}

/** Offset of the list's top inside the scroll element's content. */
export function listOffset(scroller: HTMLElement, list: HTMLElement): number {
  return Math.round(
    list.getBoundingClientRect().top -
      scroller.getBoundingClientRect().top +
      scroller.scrollTop,
  );
}

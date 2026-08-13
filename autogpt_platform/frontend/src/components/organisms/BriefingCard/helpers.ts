import { format, isToday, parseISO } from "date-fns";

export const COLLAPSED_ROWS = 3;

// Roughly six rows: enough that "show all" feels like it opened something,
// short enough that the card still sits under the composer.
const MAX_EXPANDED_HEIGHT = 416;

// Sub-pixel scroll offsets are routine (zoom, fractional row heights), so an
// exact comparison would leave both arrows on forever.
const EDGE_TOLERANCE = 2;

// Null when there is nothing to measure yet, so callers can leave the last
// good height in place rather than collapsing the card to zero.
export function measureListHeight(
  list: HTMLElement,
  isShowingAll: boolean,
): number | null {
  const rows = Array.from(list.children) as HTMLElement[];
  if (rows.length === 0) return null;
  if (isShowingAll) return Math.min(list.scrollHeight, MAX_EXPANDED_HEIGHT);

  const lastVisible = rows[Math.min(COLLAPSED_ROWS, rows.length) - 1];
  return lastVisible.offsetTop + lastVisible.offsetHeight - rows[0].offsetTop;
}

export function getScrollEdges(list: HTMLElement) {
  const { scrollTop, scrollHeight, clientHeight } = list;
  return {
    canScrollUp: scrollTop > EDGE_TOLERANCE,
    canScrollDown: scrollTop + clientHeight < scrollHeight - EDGE_TOLERANCE,
  };
}

export function formatBriefingDate(date: Date | string): string {
  // The generated type says Date, but date-only strings ("2026-08-07") skip
  // the client's date transformer (its regex requires a time part), so the
  // runtime value is still a string. parseISO reads it as LOCAL midnight —
  // `new Date("2026-08-07")` would read UTC midnight and shift the label a
  // day back for viewers west of UTC.
  const parsed = typeof date === "string" ? parseISO(date) : date;
  if (isToday(parsed)) return "This morning";
  return format(parsed, "MMMM d");
}

export function isInternalLink(link: string): boolean {
  return link.startsWith("/") && !link.startsWith("//");
}

// Relative paths only: the backend composes these, but nothing else stops a
// future regression from delivering an absolute or `javascript:` URL to a
// Next.js <Link>.
export function getSafeLink(link: string | null | undefined): string | null {
  return link && isInternalLink(link) ? link : null;
}

// Shared expand/collapse motion for the tool accordions (chain panel + row
// result panel). Grid 0fr→1fr carries the height, opacity softens the edges,
// and easeOutQuint gives the long decelerating settle.
export const ACCORDION_PANEL =
  "grid transition-[grid-template-rows,opacity] duration-400 ease-out-quint motion-reduce:transition-none";

export function accordionState(open: boolean): string {
  return open
    ? "[grid-template-rows:1fr] opacity-100"
    : "[grid-template-rows:0fr] opacity-0";
}

export const PANEL_REVEAL = "animate-fade-up motion-reduce:animate-none";

const STAGGER_MS = 45;
const MAX_STAGGERED_ROWS = 8;

// Rows cascade in behind the panel. Capped so a long chain doesn't turn the
// expand into a several-second reveal.
export function rowStaggerDelay(index: number): string {
  return `${Math.min(index, MAX_STAGGERED_ROWS) * STAGGER_MS}ms`;
}

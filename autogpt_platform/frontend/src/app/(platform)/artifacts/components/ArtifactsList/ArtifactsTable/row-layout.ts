import type { Variants } from "framer-motion";

// Column template shared by the header and every row so cells line up.
// Modified and Size collapse on narrow screens; the trailing column sizes to
// the row actions.
export const ROW_GRID_CLASS =
  "grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 sm:grid-cols-[minmax(0,1fr)_7rem_auto] md:grid-cols-[minmax(0,1fr)_8rem_6rem_auto]";
export const DATE_CELL_CLASS = "hidden truncate sm:block";
export const SIZE_CELL_CLASS = "hidden tabular-nums md:block";
export const ACTIONS_CELL_CLASS = "flex min-w-10 justify-end";

// Sized to its content (its wrapper is not stretched across the column) so
// the hover preview, which anchors to this button, opens beside the name.
export const NAME_BUTTON_CLASS =
  "flex min-w-0 items-center gap-4 rounded-xl py-2.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-zinc-400";

// Rows animate on their own mount (not via the list's orchestration) so a row
// added by an upload or a refetch is never left in its hidden start state.
// The delay grows with position, capped so appended pages don't wait long.
export const STAGGER_CAP = 8;
export const STAGGER_STEP_S = 0.03;

export const ROW_VARIANTS: Variants = {
  hidden: { opacity: 0, y: 6 },
  show: (index: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.3,
      ease: [0.16, 1, 0.3, 1],
      delay: Math.min(index, STAGGER_CAP) * STAGGER_STEP_S,
    },
  }),
};

export const REDUCED_ROW_VARIANTS: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: 0.2 } },
};

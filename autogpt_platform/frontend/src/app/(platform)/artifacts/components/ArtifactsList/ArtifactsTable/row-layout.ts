import type { Variants } from "framer-motion";

// Column template shared by the header and every row so cells line up.
// Modified and Size collapse on narrow screens; the trailing column sizes to
// the row actions.
export const ROW_GRID_CLASS =
  "grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 sm:grid-cols-[minmax(0,1fr)_7rem_auto] md:grid-cols-[minmax(0,1fr)_8rem_6rem_auto]";
export const DATE_CELL_CLASS = "hidden truncate sm:block";
export const SIZE_CELL_CLASS = "hidden tabular-nums md:block";
export const ACTIONS_CELL_CLASS = "flex min-w-10 justify-end";

export const NAME_BUTTON_CLASS =
  "flex min-w-0 items-center gap-4 rounded-xl py-2.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-zinc-400";

export const ROW_VARIANTS: Variants = {
  hidden: { opacity: 0, y: 6 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] },
  },
};

export const REDUCED_ROW_VARIANTS: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: 0.2 } },
};

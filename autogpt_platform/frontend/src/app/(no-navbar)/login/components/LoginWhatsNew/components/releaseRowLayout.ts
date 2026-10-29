// The release list and its loading skeleton share this geometry, so the
// panel keeps its height when the releases arrive.

export const RELEASE_LIST_CLASS =
  "flex flex-col gap-1 rounded-2xl bg-surface p-1.5 smooth-shadow-ring-sm";

export const RELEASE_ROW_CLASS = "flex gap-3 px-3.5 py-2.5";

export const RELEASE_DOT_COLUMN_CLASS =
  "flex w-2.5 shrink-0 justify-center pt-1";

// Spacing above the headline and the details.
export const RELEASE_HEADLINE_GAP_CLASS = "mt-1";
export const RELEASE_DETAILS_GAP_CLASS = "mt-0.5";

// Line boxes of the Text variants the row uses: the date line
// (small-medium-tight, 18px with the "Latest" pill), the headline
// (body-medium) and one line of details (body-tight).
export const DATE_LINE_HEIGHT_CLASS = "h-4";
export const LATEST_DATE_LINE_HEIGHT_CLASS = "h-[1.125rem]";
export const HEADLINE_LINE_HEIGHT_CLASS = "h-[1.375rem]";
export const DETAILS_LINE_HEIGHT_CLASS = "h-5";

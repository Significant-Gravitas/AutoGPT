import { cn } from "@/lib/utils";

const SELECTABLE_CARD_CLASS_NAME =
  "rounded-2xl border border-border bg-background p-5 transition-colors focus-within:ring-2";

export interface SelectableCardColors {
  /** Border, fill and ring once the option is chosen. */
  selected?: string;
  /** Hover border and focus ring; defaults lean on neutral tokens. */
  interactive?: string;
  /** Fill for a chosen row inside a grouped list, which has no border. */
  selectedRow?: string;
}

// The grouped card and its rows share one radius: the end rows have to round
// with the shell, otherwise their fill and focus ring get clipped square
// against its corners.
const GROUPED_RADIUS = "rounded-[2rem]";

export const groupedCardClassName = cn(
  "divide-y divide-border overflow-hidden border border-border bg-background",
  GROUPED_RADIUS,
);

// Options grouped in one card: the divider does the separating, so a row only
// carries its own fill.
export function selectableRowClassName(
  isSelected: boolean,
  colors: SelectableCardColors = {},
) {
  return cn(
    "block w-full p-5 transition-colors",
    "first:rounded-t-[2rem] last:rounded-b-[2rem]",
    // The radio is visually hidden, so this ring is the only focus cue a
    // keyboard user gets. Keyed to :focus-visible, not :focus-within, so a
    // click leaves no lingering outline.
    "has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-inset has-[:focus-visible]:ring-ring",
    isSelected
      ? (colors.selectedRow ?? "bg-accent/5")
      : "cursor-pointer hover:bg-muted/40",
  );
}

export function selectableCardClassName(
  isSelected: boolean,
  interactive = false,
  colors: SelectableCardColors = {},
) {
  return cn(
    SELECTABLE_CARD_CLASS_NAME,
    colors.interactive ?? "focus-within:ring-ring",
    isSelected
      ? (colors.selected ?? "border-accent bg-accent/5 ring-2 ring-accent/20")
      : interactive &&
          cn(
            "cursor-pointer",
            colors.interactive ? null : "hover:border-foreground/30",
          ),
  );
}

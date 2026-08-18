import { cn } from "@/lib/utils";

const SELECTABLE_CARD_CLASS_NAME =
  "rounded-2xl border border-border bg-background p-5 transition-colors has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-ring";

export interface SelectableCardColors {
  /** Border, fill and ring once the option is chosen. */
  selected?: string;
  /** Hover border and focus ring; defaults lean on neutral tokens. */
  interactive?: string;
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

import { cn } from "@/lib/utils";

const SELECTABLE_CARD_CLASS_NAME =
  "rounded-2xl border border-border bg-background p-5 transition-colors focus-within:ring-2 focus-within:ring-ring";

export function selectableCardClassName(
  isSelected: boolean,
  interactive = false,
) {
  return cn(
    SELECTABLE_CARD_CLASS_NAME,
    isSelected
      ? "border-accent bg-accent/5 ring-2 ring-accent/20"
      : interactive && "cursor-pointer hover:border-foreground/30",
  );
}

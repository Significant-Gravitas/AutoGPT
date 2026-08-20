"use client";

import { cn } from "@/lib/utils";

interface Props {
  title: string;
  subtitle?: string;
  notes?: string[];
  isSelected: boolean;
  onSelect: () => void;
  /**
   * The accessible name, when the visible title alone would not identify the
   * choice — a tier reads as "Balanced · sonnet-5" while showing the model on
   * its own line, so the name a screen reader announces stays complete.
   */
  label?: string;
}

export function ChoiceRow({
  title,
  subtitle,
  notes,
  isSelected,
  onSelect,
  label,
}: Props) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={isSelected}
      aria-label={label}
      onClick={onSelect}
      className={cn(
        "flex w-full items-start gap-2.5 px-3 py-2 text-left transition-colors",
        "focus-visible:bg-muted focus-visible:outline-none",
        isSelected ? "bg-muted/60" : "hover:bg-muted/40",
      )}
    >
      <span
        aria-hidden
        className={cn(
          "mt-[3px] flex h-3.5 w-3.5 flex-none items-center justify-center rounded-full border",
          isSelected ? "border-primary" : "border-muted-foreground/50",
        )}
      >
        {isSelected && (
          <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />
        )}
      </span>
      <span className="flex min-w-0 flex-col">
        <span className="text-xs font-medium text-foreground">{title}</span>
        {subtitle && (
          <span className="break-words text-[11px] leading-snug text-muted-foreground">
            {subtitle}
          </span>
        )}
        {notes?.map((note) => (
          <span
            key={note}
            className="mt-0.5 text-[11px] leading-snug text-muted-foreground/80"
          >
            {note}
          </span>
        ))}
      </span>
    </button>
  );
}

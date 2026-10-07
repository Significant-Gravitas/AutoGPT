import { cva } from "class-variance-authority";

// The one field style. Input, Textarea, Select, SearchInput, TimeInput,
// DateInput and DateTimeInput all start from it, and put the consumer's
// className last.
export const FIELD_BASE =
  "focus-ring w-full rounded-lg border border-input bg-background font-sans font-normal text-foreground shadow-none transition-colors placeholder:font-normal placeholder:text-zinc-500 focus-visible:ring-offset-0 disabled:cursor-not-allowed disabled:opacity-50";

export const FIELD_INVALID =
  "border-destructive focus-visible:ring-destructive";

export type FieldSize = "sm" | "md" | "lg";

// Single-line fields: 32, 36 and 40px.
export const fieldVariants = cva(FIELD_BASE, {
  variants: {
    size: {
      sm: "h-8 px-3 text-xs",
      md: "h-9 px-3 text-sm",
      lg: "h-10 px-4 text-sm",
    },
    invalid: {
      true: FIELD_INVALID,
      false: "",
    },
  },
  defaultVariants: {
    size: "md",
    invalid: false,
  },
});

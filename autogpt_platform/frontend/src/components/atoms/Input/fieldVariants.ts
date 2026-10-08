import { cva } from "class-variance-authority";

export type FieldSize = "sm" | "md" | "lg";

// House control heights (32, 36 and 40px) layered on Kobra's field primitives,
// which default to 32px. Input, Select and SearchInput read from here.
export const fieldSizeClasses: Record<FieldSize, string> = {
  sm: "h-8 px-3 text-xs md:text-xs",
  md: "h-9 px-3 text-sm",
  lg: "h-10 px-4 text-sm",
};

// Pre-Kobra field style, still rendered by DateInput, DateTimeInput and
// TimeInput until they move onto Kobra. Delete with the last importer.
// For plain <input>/<button> only: Kobra's ui/input and ui/select bring their
// own `focus-field`, so these classes on them would double the focus ring.
export const FIELD_BASE =
  "focus-ring w-full rounded-lg border border-input bg-background font-sans font-normal text-foreground shadow-none transition-colors placeholder:font-normal placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50";

export const FIELD_INVALID =
  "border-destructive focus-visible:ring-destructive";

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

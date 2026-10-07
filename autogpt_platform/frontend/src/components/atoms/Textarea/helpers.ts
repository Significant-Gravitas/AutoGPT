import { cva } from "class-variance-authority";
import { FIELD_BASE, FIELD_INVALID } from "../Input/fieldVariants";

export const textareaVariants = cva(
  [FIELD_BASE, "block resize-y text-sm leading-snug"],
  {
    variants: {
      size: {
        sm: "min-h-16 px-3 py-2",
        md: "min-h-20 px-4 py-2.5",
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
  },
);

export function getDescribedBy(
  ...ids: Array<string | undefined>
): string | undefined {
  const joined = ids.filter(Boolean).join(" ");
  return joined || undefined;
}

export function getLength(value: unknown): number {
  if (typeof value === "string") return value.length;
  if (typeof value === "number") return String(value).length;
  return 0;
}

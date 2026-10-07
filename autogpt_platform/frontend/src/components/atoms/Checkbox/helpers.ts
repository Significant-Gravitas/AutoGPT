import { cva } from "class-variance-authority";

export const checkboxVariants = cva(
  [
    "group peer inline-flex shrink-0 items-center justify-center border bg-background text-primary-foreground focus-ring transition-colors focus-visible:ring-offset-2",
    "data-[state=checked]:border-primary data-[state=checked]:bg-primary",
    "data-[state=indeterminate]:border-primary data-[state=indeterminate]:bg-primary",
    "disabled:cursor-not-allowed disabled:opacity-50",
  ],
  {
    variants: {
      size: {
        sm: "size-4 rounded-sm",
        md: "size-5 rounded-md",
      },
      invalid: {
        true: "border-destructive",
        false: "border-input hover:border-ring",
      },
    },
    defaultVariants: {
      size: "sm",
      invalid: false,
    },
  },
);

export const indicatorSize = {
  sm: 12,
  md: 14,
} as const;

export function getDescribedBy(
  ...ids: Array<string | undefined>
): string | undefined {
  const joined = ids.filter(Boolean).join(" ");
  return joined || undefined;
}

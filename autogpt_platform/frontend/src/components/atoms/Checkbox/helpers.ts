import { cva } from "class-variance-authority";

export const checkboxVariants = cva(
  [
    "group peer inline-flex shrink-0 items-center justify-center border bg-white text-white transition-colors",
    "hover:border-zinc-400",
    "focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2 focus-visible:outline-hidden",
    "data-[state=checked]:border-zinc-800 data-[state=checked]:bg-zinc-800",
    "data-[state=indeterminate]:border-zinc-800 data-[state=indeterminate]:bg-zinc-800",
    "disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:border-zinc-300",
  ],
  {
    variants: {
      size: {
        sm: "size-4 rounded-sm",
        md: "size-5 rounded-md",
      },
      invalid: {
        true: "border-red-500 hover:border-red-500",
        false: "border-zinc-300",
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

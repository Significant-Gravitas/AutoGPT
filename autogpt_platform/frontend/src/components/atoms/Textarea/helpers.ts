import { cva } from "class-variance-authority";

export const textareaVariants = cva(
  [
    "block w-full resize-y rounded-xl border bg-white font-sans text-sm font-normal leading-snug text-black shadow-none transition-colors",
    "placeholder:font-normal placeholder:text-zinc-500",
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2",
    "disabled:cursor-not-allowed disabled:opacity-50",
  ],
  {
    variants: {
      size: {
        sm: "min-h-16 px-3 py-2",
        md: "min-h-20 px-4 py-2.5",
      },
      invalid: {
        true: "border-red-500 focus-visible:ring-red-500",
        false: "border-zinc-200 hover:border-zinc-300",
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

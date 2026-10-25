import { cn } from "@/lib/utils";

interface Props extends React.HTMLAttributes<HTMLDivElement> {
  className?: string;
  /** `translucent` is the dark wash of the deleted components/ui/skeleton,
   *  which takes its tint from whatever sits behind it. */
  variant?: "solid" | "translucent";
}

// TODO(tokens): no role holds neutral-900 #171717, the translucent wash. Its
// dark: class comes verbatim from the deleted components/ui/skeleton.
const variantClassName = {
  solid: "bg-surface-hover",
  translucent: "bg-neutral-900/10 dark:bg-neutral-50/10",
} as const;

export function Skeleton({ className, variant = "solid", ...props }: Props) {
  return (
    <div
      data-slot="skeleton"
      className={cn(
        "animate-pulse rounded-md",
        variantClassName[variant],
        className,
      )}
      {...props}
    />
  );
}

import { cn } from "@/lib/utils";

interface Props extends React.HTMLAttributes<HTMLDivElement> {
  className?: string;
  /** `translucent` is the dark wash of the deleted components/ui/skeleton,
   *  which takes its tint from whatever sits behind it. `nested` is a block
   *  drawn on a solid skeleton card, a step darker so it shows. */
  variant?: "solid" | "translucent" | "nested";
}

const variantClassName = {
  solid: "bg-surface-hover",
  translucent: "bg-surface-inverse/10",
  nested: "bg-surface-active",
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

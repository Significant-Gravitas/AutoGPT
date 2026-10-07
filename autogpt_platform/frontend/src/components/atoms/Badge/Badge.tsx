import { cn } from "@/lib/utils";

type BadgeVariant = "success" | "error" | "warning" | "info";
type BadgeSize = "small" | "medium";

interface BadgeProps {
  variant: BadgeVariant;
  size?: BadgeSize;
  children: React.ReactNode;
  className?: string;
}

// Fill and ring from the semantic tokens. Text uses the 700/800 step of the
// same hue: the token colours are below 4.5:1 on their own tint.
const badgeVariants: Record<BadgeVariant, string> = {
  success: "bg-success/10 text-green-700 ring-success/20",
  error: "bg-destructive/10 text-red-700 ring-destructive/20",
  warning: "bg-warning/10 text-yellow-800 ring-warning/30",
  info: "bg-info-foreground text-info ring-info/10",
};

const badgeSizes: Record<BadgeSize, string> = {
  small: "px-1.5 py-0.5 text-[11px] leading-4",
  medium: "px-2 py-0.5 text-xs leading-5",
};

export function Badge({
  variant,
  size = "medium",
  children,
  className,
}: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex max-w-full items-center gap-1.5 rounded-md font-sans font-medium ring-1 ring-inset",
        badgeSizes[size],
        badgeVariants[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}

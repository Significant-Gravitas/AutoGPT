import { cn } from "@/lib/utils";

type BadgeVariant = "success" | "error" | "warning" | "info";
type BadgeSize = "small" | "medium";

interface BadgeProps {
  variant: BadgeVariant;
  size?: BadgeSize;
  children: React.ReactNode;
  className?: string;
}

const badgeVariants: Record<BadgeVariant, string> = {
  success: "bg-green-50 text-green-700 ring-green-600/20",
  error: "bg-red-50 text-red-700 ring-red-600/10",
  warning: "bg-yellow-50 text-yellow-800 ring-yellow-500/20",
  info: "bg-zinc-50 text-zinc-600 ring-zinc-500/10",
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
        "inline-flex items-center gap-1.5 rounded-md font-sans font-medium ring-1 ring-inset",
        "overflow-hidden",
        badgeSizes[size],
        badgeVariants[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}

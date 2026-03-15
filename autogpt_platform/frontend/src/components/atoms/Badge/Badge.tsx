import { cn } from "@/lib/utils";

type BadgeVariant = "success" | "error" | "info";
type BadgeSize = "small" | "medium";

interface BadgeProps {
  variant: BadgeVariant;
  size?: BadgeSize;
  children: React.ReactNode;
  className?: string;
}

const badgeVariants: Record<BadgeVariant, string> = {
  success: "bg-green-100 text-green-800",
  error: "bg-red-100 text-red-800",
  info: "bg-slate-50 text-black",
};

const badgeSizes: Record<BadgeSize, string> = {
  small: "px-[6px] py-[3px] text-[0.55rem] leading-4 tracking-widest",
  medium: "px-[9px] py-[3px] text-[0.6785rem] leading-5 tracking-wider",
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
        // Base styles from Figma
        "inline-flex items-center gap-2 rounded-[45px]",
        // Text styles
        "font-sans font-medium uppercase",
        // Text overflow handling
        "overflow-hidden text-ellipsis",
        // Size styles
        badgeSizes[size],
        // Variant styles
        badgeVariants[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}

import { Badge as KobraBadge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type BadgeVariant = "success" | "error" | "warning" | "info";
type BadgeSize = "small" | "medium";

interface BadgeProps {
  variant: BadgeVariant;
  size?: BadgeSize;
  children: React.ReactNode;
  className?: string;
}

// The house `info` badge is the neutral zinc one (Stopped, Running), not blue.
const badgeVariants: Record<
  BadgeVariant,
  React.ComponentProps<typeof KobraBadge>["variant"]
> = {
  success: "green",
  error: "destructive",
  warning: "amber",
  info: "neutral",
};

// Kobra's sizes are 20/24px like the house ones; the house keeps its smaller type.
const badgeSizes: Record<
  BadgeSize,
  { size: React.ComponentProps<typeof KobraBadge>["size"]; className: string }
> = {
  small: { size: "sm", className: "text-[11px]" },
  medium: { size: "default", className: "text-xs" },
};

export function Badge({
  variant,
  size = "medium",
  children,
  className,
}: BadgeProps) {
  return (
    <KobraBadge
      variant={badgeVariants[variant]}
      size={badgeSizes[size].size}
      className={cn(
        "max-w-full gap-1.5",
        badgeSizes[size].className,
        className,
      )}
    >
      {children}
    </KobraBadge>
  );
}

import { cn } from "@/lib/utils";

type BadgeVariant = "success" | "error" | "info";

interface BadgeProps {
  variant: BadgeVariant;
  children: React.ReactNode;
  className?: string;
}

const badgeVariants: Record<BadgeVariant, string> = {
  success: "bg-green-100 text-green-800",
  error: "bg-red-100 text-red-800",
  info: "bg-slate-50 text-black",
};

export function Badge({ variant, children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        // Base styles from Figma
        "inline-flex items-center gap-2 rounded-[45px] px-[9px] py-[3px]",
        // Text styles
        "font-sans text-[0.6785rem] font-medium uppercase leading-5 tracking-wider",
        // Text overflow handling
        "overflow-hidden text-ellipsis",
        // Variant styles
        badgeVariants[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}

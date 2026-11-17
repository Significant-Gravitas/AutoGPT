import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import type { IconSvgElement } from "@hugeicons/react";
import type { ReactNode } from "react";

interface Props {
  tone: "success" | "warning";
  icon: IconSvgElement;
  children: ReactNode;
}

export function StatusTitle({ tone, icon, children }: Props) {
  return (
    <span className="flex items-center gap-3">
      <span
        aria-hidden
        className={cn(
          "flex size-9 shrink-0 items-center justify-center rounded-full",
          tone === "success"
            ? "bg-success-surface text-success-foreground"
            : "bg-warning-surface text-warning-foreground",
        )}
      >
        <Icon icon={icon} size={18} />
      </span>
      <span className="min-w-0">{children}</span>
    </span>
  );
}

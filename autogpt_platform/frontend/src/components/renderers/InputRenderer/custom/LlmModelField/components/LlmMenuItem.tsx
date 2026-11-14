"use client";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowRight01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type Props = {
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  showChevron?: boolean;
  rightSlot?: React.ReactNode;
  onClick: () => void;
  isActive?: boolean;
};

export function LlmMenuItem({
  title,
  subtitle,
  icon,
  showChevron,
  rightSlot,
  onClick,
  isActive,
}: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn("w-full py-1 pr-4 pl-2 text-left hover:bg-surface-hover")}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {icon}
          <Text variant="body" tone="primary">
            {title}
          </Text>
        </div>
        <div className="flex items-center gap-2">
          {isActive && (
            <Icon icon={Tick02Icon} tone="success" className="h-4 w-4" />
          )}
          {rightSlot}
          {showChevron && (
            <Icon icon={ArrowRight01Icon} tone="primary" className="h-4 w-4" />
          )}
        </div>
      </div>
      {subtitle && (
        <Text variant="small" tone="muted" className="mb-1">
          {subtitle}
        </Text>
      )}
    </button>
  );
}

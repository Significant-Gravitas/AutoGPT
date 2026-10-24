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
  const hasIcon = Boolean(icon);

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn("w-full py-1 pl-2 pr-4 text-left hover:bg-surface-hover")}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {icon}
          <Text variant="body" className="text-foreground-strong">
            {title}
          </Text>
        </div>
        <div className="flex items-center gap-2">
          {isActive && (
            <Icon
              // TODO(tokens): no exact role for text-emerald-600 #059669
              icon={Tick02Icon}
              className="h-4 w-4 text-emerald-600"
            />
          )}
          {rightSlot}
          {showChevron && (
            <Icon
              icon={ArrowRight01Icon}
              className="h-4 w-4 text-foreground-strong"
            />
          )}
        </div>
      </div>
      {subtitle && (
        <Text
          variant="small"
          className={cn("mb-1 text-foreground-tertiary", hasIcon && "pl-0")}
        >
          {subtitle}
        </Text>
      )}
    </button>
  );
}

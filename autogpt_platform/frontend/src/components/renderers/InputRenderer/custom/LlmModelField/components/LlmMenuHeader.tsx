"use client";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type Props = {
  label: string;
  onBack: () => void;
};

export function LlmMenuHeader({ label, onBack }: Props) {
  return (
    <button
      type="button"
      onClick={onBack}
      className="flex w-full items-center gap-2 px-2 py-2 text-left hover:bg-surface-hover"
    >
      <Icon
        icon={ArrowLeft02Icon}
        className="h-4 w-4 text-foreground-emphasis"
      />
      <Text variant="body" className="text-foreground-strong">
        {label}
      </Text>
    </button>
  );
}

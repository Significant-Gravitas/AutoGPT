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
      className="flex w-full items-center gap-2 px-2 py-2 text-left hover:bg-zinc-100"
    >
      <Icon icon={ArrowLeft02Icon} className="h-4 w-4 text-zinc-800" />
      <Text variant="body" className="text-zinc-900">
        {label}
      </Text>
    </button>
  );
}

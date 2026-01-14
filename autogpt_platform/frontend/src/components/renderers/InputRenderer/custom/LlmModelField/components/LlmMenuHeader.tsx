"use client";

import { ArrowLeftIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";

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
      <ArrowLeftIcon className="h-4 w-4 text-zinc-800" weight="bold" />
      <Text variant="body" className="text-zinc-900">
        {label}
      </Text>
    </button>
  );
}

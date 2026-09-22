"use client";

import { DollarSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type Props = {
  tier?: number;
};

export function LlmPriceTier({ tier }: Props) {
  if (!tier || tier <= 0) {
    return null;
  }
  const clamped = Math.min(3, Math.max(1, tier));
  return (
    <div className="flex items-center text-zinc-900">
      {Array.from({ length: clamped }).map((_, index) => (
        <Icon
          icon={DollarSignIcon}
          key={`price-${index}`}
          className="-mr-0.5 h-3 w-3"
        />
      ))}
    </div>
  );
}

"use client";

import { CurrencyDollarSimpleIcon } from "@phosphor-icons/react";

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
        <CurrencyDollarSimpleIcon
          key={`price-${index}`}
          className="-mr-0.5 h-3 w-3"
          weight="bold"
        />
      ))}
    </div>
  );
}

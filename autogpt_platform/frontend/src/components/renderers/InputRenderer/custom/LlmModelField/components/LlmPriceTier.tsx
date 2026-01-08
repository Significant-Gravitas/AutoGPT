"use client";

type Props = {
  tier?: number;
};

export function LlmPriceTier({ tier }: Props) {
  if (!tier || tier <= 0) {
    return null;
  }
  const clamped = Math.min(3, Math.max(1, tier));
  return (
    <div className="flex items-center gap-0.5 text-xs text-zinc-600">
      {Array.from({ length: clamped }).map((_, index) => (
        <span key={`price-${index}`}>$</span>
      ))}
    </div>
  );
}

import { ArrowRight02Icon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";
import { formatUpgradePrice, getMaxUpgradePricing } from "./maxUpgrade";

type Props = { pricing: ReturnType<typeof getMaxUpgradePricing> };

export function MaxUpgradePricing({ pricing }: Props) {
  if (!pricing?.maxCents) return null;
  const period = pricing.cycle === "yearly" ? "year" : "month";
  const delta =
    pricing.currentCents === null
      ? null
      : pricing.maxCents - pricing.currentCents;
  return (
    <div className="border-t border-zinc-200 pt-5">
      <div className="flex flex-wrap items-center gap-4">
        {pricing.currentCents !== null && (
          <>
            <div className="min-w-0 flex-1 text-zinc-600">
              <p className="mb-1 text-xs">Pro · your plan</p>
              <p className="text-xl tracking-tight">
                {formatUpgradePrice(pricing.currentCents)}
                <span className="text-xs tracking-normal"> / {period}</span>
              </p>
            </div>
            <Icon
              icon={ArrowRight02Icon}
              size={18}
              className="flex-none text-zinc-500"
              aria-hidden
            />
          </>
        )}
        <div className="min-w-0 flex-1 text-zinc-900">
          <p className="mb-1 text-xs font-medium">Max</p>
          <p className="text-2xl font-medium tracking-tight">
            {formatUpgradePrice(pricing.maxCents)}
            <span className="text-xs font-normal tracking-normal text-zinc-600">
              {" "}
              / {period}
            </span>
          </p>
        </div>
      </div>
      {delta !== null && delta > 0 && (
        <p className="mt-3 text-xs text-zinc-600">
          {formatUpgradePrice(delta)} more per {period} · billed {pricing.cycle}
        </p>
      )}
      <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">
        Standard plan prices, before taxes and any applicable discounts.
      </p>
    </div>
  );
}

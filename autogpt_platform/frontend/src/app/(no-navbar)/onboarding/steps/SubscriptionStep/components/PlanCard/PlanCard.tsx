"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Check, Info, Star } from "@phosphor-icons/react";
import { type Country, formatPrice } from "../../countries";
import { type PlanDef, type PlanKey } from "../../helpers";
import { computePlanPricing } from "./helpers";

interface Props {
  plan: PlanDef;
  country: Country;
  isYearly: boolean;
  onSelect: (key: PlanKey) => void;
  className?: string;
}

export function PlanCard({
  plan,
  country,
  isYearly,
  onSelect,
  className,
}: Props) {
  const { displayPrice, monthlyEquiv, perLabel } = computePlanPricing({
    plan,
    country,
    isYearly,
  });
  const hl = plan.highlighted;

  return (
    <div
      className={cn(
        "relative rounded-2xl p-px",
        hl
          ? "bg-gradient-to-br from-purple-500 via-fuchsia-500 to-indigo-500 shadow-[0_10px_32px_-8px_rgba(119,51,245,0.25)]"
          : "bg-gradient-to-br from-zinc-300 via-zinc-400 to-zinc-500 md:my-5",
        className,
      )}
    >
      {plan.badge && hl && (
        <span className="absolute -top-3 right-5 z-10 inline-flex overflow-hidden rounded-full p-px shadow-[0_10px_28px_-6px_rgba(124,58,237,0.55)]">
          <span
            aria-hidden
            className="absolute -inset-[150%] animate-[spin_4s_linear_infinite] bg-[conic-gradient(from_0deg,#a855f7,#7c3aed,#4f46e5,#1e40af,#4f46e5,#7c3aed,#a855f7)]"
          />
          <span className="relative inline-flex items-center gap-1 rounded-full bg-gradient-to-r from-indigo-600 via-blue-600 to-blue-500 px-2.5 py-1 text-[10px] font-semibold text-white">
            <Star
              size={10}
              weight="fill"
              aria-hidden="true"
              className="text-yellow-300"
            />
            {plan.badge}
          </span>
        </span>
      )}

      <div
        className={cn(
          "relative flex h-full flex-col overflow-hidden rounded-[15px] bg-white",
          hl ? "p-6" : "p-5",
        )}
      >
        <div
          aria-hidden
          className={cn(
            "pointer-events-none absolute inset-0",
            hl
              ? "bg-[radial-gradient(120%_60%_at_0%_0%,rgba(168,85,247,0.10),transparent_60%),radial-gradient(120%_60%_at_100%_100%,rgba(99,102,241,0.08),transparent_60%)]"
              : "bg-[radial-gradient(140%_70%_at_0%_0%,rgba(244,244,245,0.7),transparent_60%)]",
          )}
        />

        <div className="relative flex h-full flex-col">
          <div className="mb-2 flex items-center gap-2">
            <Text
              variant="h5"
              as="h2"
              className="font-poppins !text-[17px] !font-semibold !text-zinc-800"
            >
              {plan.name}
            </Text>
            {plan.usage && (
              <span className="inline-flex items-center gap-1 rounded-full bg-purple-100 px-2 py-0.5 text-[10px] font-medium text-purple-700">
                {plan.usage} usage
              </span>
            )}
            {plan.badge && !hl && (
              <span className="inline-flex items-center gap-1 rounded-full bg-zinc-100 px-2 py-0.5 text-[10px] font-medium text-zinc-500">
                {plan.badge}
              </span>
            )}
          </div>

          <div className="mb-1 flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5">
            {displayPrice !== null ? (
              <>
                <span
                  className={cn(
                    "font-poppins font-medium leading-none text-zinc-800",
                    hl ? "text-[32px]" : "text-[26px]",
                  )}
                >
                  {formatPrice(
                    displayPrice,
                    country.currencyCode,
                    country.symbol,
                  )}
                </span>
                <span className="text-xs text-zinc-400">{perLabel}</span>
                {monthlyEquiv !== null && (
                  <span className="text-xs text-zinc-800">
                    (
                    {formatPrice(
                      monthlyEquiv,
                      country.currencyCode,
                      country.symbol,
                    )}
                    /month)
                  </span>
                )}
              </>
            ) : (
              <span
                className={cn(
                  "font-poppins font-medium leading-none text-zinc-800",
                  hl ? "text-[32px]" : "text-[26px]",
                )}
              >
                Contact us
              </span>
            )}
          </div>

          <p className="mb-3.5 min-h-[30px] text-sm leading-relaxed text-zinc-800">
            {plan.description}
          </p>

          <div className="mb-3 h-px bg-zinc-100" />

          <div className="mb-4 flex flex-1 flex-col gap-2">
            {plan.features.map((f) => (
              <div key={f} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span
                    aria-hidden="true"
                    className="flex h-4 w-4 items-center justify-center rounded-full bg-purple-50"
                  >
                    <Check
                      size={10}
                      weight="bold"
                      className="text-purple-500"
                    />
                  </span>
                  <span className="text-[13px] text-zinc-800">{f}</span>
                </div>
                <Info size={13} className="text-zinc-300" aria-hidden="true" />
              </div>
            ))}
          </div>

          <Button
            variant={plan.buttonVariant}
            size={hl ? "large" : "small"}
            onClick={() => onSelect(plan.key)}
            className="w-full"
          >
            {plan.cta}
          </Button>
        </div>
      </div>
    </div>
  );
}

"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { CountrySelector } from "./components/CountrySelector/CountrySelector";
import { PlanCard } from "./components/PlanCard/PlanCard";
import { PLAN_KEYS, PLANS } from "./helpers";
import { useSubscriptionStep } from "./useSubscriptionStep";

export function SubscriptionStep() {
  const {
    billing,
    setBilling,
    countryIdx,
    setCountryIdx,
    country,
    isYearly,
    handlePlanSelect,
  } = useSubscriptionStep();

  return (
    <FadeIn>
      <div className="-mt-[2.2rem] flex w-full flex-col items-center gap-4 px-4">
        <AutoGPTLogo className="relative right-[2rem] h-14 w-[9rem]" hideText />

        <div className="flex flex-col items-center gap-1 text-center">
          <Text
            variant="h3"
            className="!text-[1.375rem] !leading-[1.6rem] md:!text-[1.75rem] md:!leading-[2.5rem]"
          >
            Choose the plan that&apos;s right for{" "}
            <span className="bg-gradient-to-r from-purple-500 to-indigo-500 bg-clip-text text-transparent">
              you
            </span>
          </Text>
          <Text variant="body" className="!text-zinc-500">
            Upgrade, downgrade, or change plans anytime. All plans include core
            features to get you started.
          </Text>
        </div>

        <div className="relative flex w-full max-w-[960px] flex-col items-center md:flex-row md:justify-center">
          <div className="inline-flex rounded-full border border-[#d8d8d8] bg-zinc-100 p-[3px]">
            {(["monthly", "yearly"] as const).map((cycle) => (
              <button
                key={cycle}
                type="button"
                onClick={() => setBilling(cycle)}
                className={cn(
                  "rounded-full border-none px-4 py-1.5 text-xs font-medium transition-all",
                  billing === cycle
                    ? "bg-white text-zinc-900 shadow-sm"
                    : "bg-transparent text-zinc-500 hover:text-zinc-700",
                )}
              >
                {cycle === "monthly" ? (
                  "Monthly billing"
                ) : (
                  <>
                    Yearly billing{" "}
                    <span className="ml-1.5 bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500 bg-clip-text text-[11px] font-semibold text-transparent">
                      Save 15%
                    </span>
                  </>
                )}
              </button>
            ))}
          </div>
          <div className="mb-0 mt-4 md:absolute md:right-0 md:my-0">
            <CountrySelector selected={countryIdx} onSelect={setCountryIdx} />
          </div>
        </div>

        <div className="mt-2 grid w-full max-w-[960px] grid-cols-1 gap-4 px-[1rem] md:grid-cols-3 md:px-0">
          {PLANS.map((plan) => (
            <PlanCard
              key={plan.key}
              plan={plan}
              country={country}
              isYearly={isYearly}
              onSelect={handlePlanSelect}
              className={cn(
                plan.key === PLAN_KEYS.MAX && "order-1 md:order-none",
                plan.key === PLAN_KEYS.PRO && "order-2 md:order-none",
                plan.key === PLAN_KEYS.TEAM && "order-3 md:order-none",
              )}
            />
          ))}
        </div>

        <Text variant="body" className="!text-zinc-500">
          Need something custom?{" "}
          <a
            href="mailto:sales@agpt.co"
            className="font-medium text-purple-500 no-underline hover:text-purple-600"
          >
            Talk to sales.
          </a>
        </Text>
      </div>
    </FadeIn>
  );
}

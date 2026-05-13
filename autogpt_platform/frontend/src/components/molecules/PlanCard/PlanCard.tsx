"use client";

import NumberFlow from "@number-flow/react";
import { motion, useReducedMotion } from "framer-motion";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { CheckIcon, StarIcon } from "@phosphor-icons/react";
import { type Country } from "./countries";
import { PLAN_KEYS, type PlanDef, type PlanKey } from "./plans";
import { computePlanPricing } from "./computePricing";

const ZERO_DECIMAL_CODES = new Set(["JPY", "KRW", "HUF", "CLP"]);

interface Props {
  plan: PlanDef;
  country: Country;
  isYearly: boolean;
  onSelect: (key: PlanKey) => void;
  className?: string;
  loading?: boolean;
  disabled?: boolean;
}

export function PlanCard({
  plan,
  country,
  isYearly,
  onSelect,
  className,
  loading = false,
  disabled = false,
}: Props) {
  const { primaryPrice, chargedToday } = computePlanPricing({
    plan,
    country,
    isYearly,
  });
  const hl = plan.highlighted;
  const isTeam = plan.key === PLAN_KEYS.TEAM;
  const reduceMotion = useReducedMotion();
  const decimalDigits = ZERO_DECIMAL_CODES.has(country.currencyCode) ? 0 : 2;
  const moneyFormat = {
    style: "currency",
    currency: country.currencyCode,
    currencyDisplay: "symbol",
    minimumFractionDigits: decimalDigits,
    maximumFractionDigits: decimalDigits,
  } as const;
  const moneyFormatter = new Intl.NumberFormat("en-US", moneyFormat);

  return (
    <div
      aria-disabled={disabled || undefined}
      className={cn(
        "relative rounded-2xl p-px transition-opacity",
        hl
          ? "bg-gradient-to-br from-purple-500 via-fuchsia-500 to-indigo-500 shadow-[0_10px_32px_-8px_rgba(119,51,245,0.25)]"
          : "bg-gradient-to-br from-zinc-300 via-zinc-400 to-zinc-500 md:my-5",
        disabled && "pointer-events-none opacity-50",
        className,
      )}
    >
      {plan.badge && hl && (
        <motion.span
          className="absolute -top-3 right-5 z-10 inline-flex overflow-hidden rounded-full p-px shadow-[0_10px_28px_-6px_rgba(124,58,237,0.55)]"
          initial={reduceMotion ? false : { scale: 0, opacity: 0, rotate: -12 }}
          animate={
            reduceMotion ? undefined : { scale: 1, opacity: 1, rotate: 0 }
          }
          transition={
            reduceMotion
              ? undefined
              : { type: "spring", stiffness: 320, damping: 14, delay: 0.45 }
          }
        >
          <span
            aria-hidden
            className="absolute -inset-[150%] animate-[spin_4s_linear_infinite] bg-[conic-gradient(from_0deg,#a855f7,#7c3aed,#4f46e5,#1e40af,#4f46e5,#7c3aed,#a855f7)]"
          />
          <span className="relative inline-flex items-center gap-1 rounded-full bg-gradient-to-r from-indigo-600 via-blue-600 to-blue-500 px-2.5 py-1 text-[10px] font-semibold text-white">
            <motion.span
              className="inline-flex"
              animate={
                reduceMotion
                  ? undefined
                  : { rotate: [0, 18, -10, 0], scale: [1, 1.25, 0.95, 1] }
              }
              transition={
                reduceMotion
                  ? undefined
                  : {
                      duration: 1.2,
                      repeat: Infinity,
                      repeatDelay: 2.2,
                      ease: "easeInOut",
                    }
              }
            >
              <StarIcon
                size={10}
                weight="fill"
                aria-hidden="true"
                className="text-yellow-300"
              />
            </motion.span>
            {plan.badge}
          </span>
        </motion.span>
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
            {primaryPrice !== null ? (
              <>
                <span
                  aria-label={moneyFormatter.format(primaryPrice)}
                  className={cn(
                    "font-poppins font-medium leading-none text-zinc-800",
                    hl ? "text-[32px]" : "text-[26px]",
                  )}
                >
                  <NumberFlow
                    value={primaryPrice}
                    format={moneyFormat}
                    locales="en-US"
                    willChange
                  />
                </span>
                <span className="text-xs text-zinc-400">/ month</span>
              </>
            ) : (
              <span
                className={cn(
                  "font-poppins font-medium leading-none tracking-[-1px] text-zinc-800",
                  hl ? "text-[32px]" : "text-[26px]",
                )}
              >
                Contact us
              </span>
            )}
          </div>

          {chargedToday !== null && (
            <div
              aria-label={`Charged today: ${moneyFormatter.format(chargedToday)}`}
              className="mb-1 flex items-baseline gap-1 text-xs font-medium text-zinc-700"
            >
              <span>Charged today:</span>
              <NumberFlow
                value={chargedToday}
                format={moneyFormat}
                locales="en-US"
                willChange
              />
            </div>
          )}

          <p className="mb-3.5 min-h-[30px] text-sm leading-relaxed text-zinc-800">
            {plan.description}
          </p>

          <div className="mb-3 h-px bg-zinc-100" />

          <div className="mb-4 flex flex-1 flex-col gap-2">
            {plan.features.map((f) => (
              <div key={f} className="flex items-center gap-2">
                <span
                  aria-hidden="true"
                  className={cn(
                    "flex h-4 w-4 items-center justify-center rounded-full",
                    isTeam ? "bg-stone-100" : "bg-purple-50",
                  )}
                >
                  <CheckIcon
                    size={10}
                    weight="bold"
                    className={isTeam ? "text-stone-500" : "text-purple-500"}
                  />
                </span>
                <span className="text-[13px] text-zinc-800">{f}</span>
              </div>
            ))}
          </div>

          <div className="relative isolate mt-5 w-full">
            {hl && (
              <motion.span
                aria-hidden
                className="pointer-events-none absolute -inset-0.5 -z-10 block rounded-[14px] bg-[linear-gradient(90deg,#a855f7,#7c3aed,#4f46e5,#6366f1,#a855f7)] bg-[length:200%_100%] opacity-25 blur-md"
                animate={
                  reduceMotion
                    ? undefined
                    : { backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"] }
                }
                transition={
                  reduceMotion
                    ? undefined
                    : { duration: 8, repeat: Infinity, ease: "linear" }
                }
              />
            )}
            <Button
              variant={plan.buttonVariant}
              size={hl ? "large" : "small"}
              onClick={() => onSelect(plan.key)}
              className="w-full"
              loading={loading}
              disabled={disabled}
            >
              {plan.cta}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

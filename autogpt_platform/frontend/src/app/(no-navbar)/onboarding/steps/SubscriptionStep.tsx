"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { cn } from "@/lib/utils";
import { Check, Info, MagnifyingGlass, Star } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import { useOnboardingWizardStore } from "../store";
import { COUNTRIES, formatPrice, type Country } from "./countries";

/* ─── Plan definitions ─── */

const PRO_USD_MONTHLY = 50;
const MAX_USD_MONTHLY = 320;

interface PlanDef {
  key: string;
  name: string;
  usage: string | null;
  usdMonthly: number | null;
  description: string;
  features: string[];
  cta: string;
  highlighted: boolean;
  badge: string | null;
  buttonVariant: "primary" | "secondary" | "outline";
}

const PLANS: PlanDef[] = [
  {
    key: "PRO",
    name: "Pro",
    usage: "1x",
    usdMonthly: PRO_USD_MONTHLY,
    description:
      "For individuals getting started with dependable everyday automation.",
    features: [
      "1x usage allowance",
      "Up to 1,000 agent runs / month",
      "Access to core models",
      "Core agent library",
      "Standard support",
    ],
    cta: "Get Pro",
    highlighted: false,
    badge: null,
    buttonVariant: "secondary",
  },
  {
    key: "BUSINESS",
    name: "Max",
    usage: "7x",
    usdMonthly: MAX_USD_MONTHLY,
    description:
      "For power users running more workflows and higher-volume automations.",
    features: [
      "7x usage allowance",
      "Up to 10,000 agent runs / month",
      "Access to premium models",
      "Advanced workflows & tools",
      "Priority processing",
      "Faster support",
      "Custom integrations (beta)",
    ],
    cta: "Upgrade to Max",
    highlighted: true,
    badge: "Best value",
    buttonVariant: "primary",
  },
  {
    key: "ENTERPRISE",
    name: "Team",
    usage: null,
    usdMonthly: null,
    description:
      "For teams that need collaboration, controls, and custom usage.",
    features: [
      "Multi-user workspaces",
      "Admin controls & roles",
      "Shared billing & usage pools",
      "Collaboration & approvals",
      "Security & compliance options",
      "Premium support",
    ],
    cta: "Contact sales",
    highlighted: false,
    badge: "Coming soon",
    buttonVariant: "outline",
  },
];

/* ─── Country Selector (opens upward) ─── */

function CountrySelector({
  selected,
  onSelect,
}: {
  selected: number;
  onSelect: (idx: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const country = COUNTRIES[selected];
  const filtered = COUNTRIES.filter(
    (c) =>
      c.name.toLowerCase().includes(search.toLowerCase()) ||
      c.code.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => {
          setOpen(!open);
          setSearch("");
        }}
        className="flex min-w-[150px] items-center gap-1.5 rounded-lg border border-zinc-300 bg-white px-2.5 py-1.5 text-xs text-zinc-600 transition-colors hover:border-zinc-400"
      >
        <span className="text-base">{country.flag}</span>
        <span>{country.name}</span>
        <span className="ml-auto text-[9px] text-zinc-400">▾</span>
      </button>

      {open && (
        <div className="absolute bottom-[calc(100%+8px)] right-0 z-50 flex max-h-[340px] w-[260px] flex-col overflow-hidden rounded-xl bg-zinc-900 shadow-xl">
          {/* Search */}
          <div className="px-3 pb-1.5 pt-2.5">
            <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5">
              <MagnifyingGlass
                size={12}
                className="shrink-0 text-white/40"
                weight="bold"
              />
              <input
                // eslint-disable-next-line jsx-a11y/no-autofocus
                autoFocus
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search country…"
                className="w-full bg-transparent text-xs text-white outline-none placeholder:text-white/35"
              />
            </div>
          </div>

          {/* List */}
          <div className="max-h-[270px] overflow-y-auto px-0 py-1">
            {filtered.length === 0 && (
              <div className="px-3.5 py-3 text-xs text-white/40">
                No matching countries
              </div>
            )}
            {filtered.map((c) => {
              const idx = COUNTRIES.indexOf(c);
              return (
                <button
                  key={c.code}
                  type="button"
                  onClick={() => {
                    onSelect(idx);
                    setOpen(false);
                  }}
                  className={cn(
                    "flex w-full items-center justify-between px-3.5 py-2 text-xs text-white transition-colors",
                    idx === selected
                      ? "bg-purple-500/15"
                      : "hover:bg-white/[0.08]",
                  )}
                >
                  <span>
                    {c.flag}&nbsp;&nbsp;{c.name}
                  </span>
                  <span className="text-[10px] text-white/35">{c.code}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Plan Card ─── */

function PlanCard({
  plan,
  country,
  isYearly,
  onSelect,
}: {
  plan: PlanDef;
  country: Country;
  isYearly: boolean;
  onSelect: (key: string) => void;
}) {
  const monthlyLocal = plan.usdMonthly
    ? plan.usdMonthly * country.rate
    : null;
  const displayPrice = monthlyLocal
    ? isYearly
      ? monthlyLocal * 0.85 * 12
      : monthlyLocal
    : null;
  const perLabel = isYearly ? "/ year" : "/ month";
  const monthlyEquiv =
    isYearly && monthlyLocal ? monthlyLocal * 0.85 : null;
  const hl = plan.highlighted;

  return (
    <div
      className={cn(
        "flex flex-col rounded-2xl bg-white",
        hl
          ? "border-2 border-purple-500 p-[21px] shadow-[0_4px_20px_rgba(119,51,245,0.07)]"
          : "border border-zinc-300 p-[22px]",
      )}
    >
      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <Text variant="h5" as="h2" className="font-poppins !text-[17px] !font-semibold">
          {plan.name}
        </Text>
        {plan.badge && (
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium",
              hl
                ? "bg-purple-50 text-purple-700"
                : "bg-zinc-100 text-zinc-500",
            )}
          >
            {hl && <Star size={10} weight="fill" className="text-purple-500" />}
            {plan.badge}
          </span>
        )}
      </div>

      {/* Usage badge */}
      {plan.usage ? (
        <span
          className={cn(
            "mb-2.5 inline-block w-fit rounded px-2 py-0.5 text-[11px] font-medium",
            hl
              ? "border border-purple-100 bg-purple-50 text-purple-700"
              : "border border-zinc-200 bg-zinc-50 text-zinc-500",
          )}
        >
          {plan.usage} usage
        </span>
      ) : (
        <div className="mb-2.5 h-[11px]" />
      )}

      {/* Price */}
      <div className="mb-1 flex items-baseline gap-1">
        {displayPrice !== null ? (
          <>
            <span className="font-poppins text-[32px] font-bold leading-none">
              {formatPrice(displayPrice, country.code, country.symbol)}
            </span>
            <span className="text-xs text-zinc-400">{perLabel}</span>
          </>
        ) : (
          <span className="font-poppins text-xl font-semibold">Contact us</span>
        )}
      </div>

      {/* Monthly equivalent (shown when yearly) */}
      {monthlyEquiv !== null && (
        <p className="mb-0.5 text-[11px] text-zinc-400">
          {formatPrice(monthlyEquiv, country.code, country.symbol)} / month
        </p>
      )}

      {/* Description */}
      <p className="mb-3.5 min-h-[30px] text-[11px] leading-relaxed text-zinc-400">
        {plan.description}
      </p>

      {/* Divider */}
      <div className="mb-3 h-px bg-zinc-100" />

      {/* Features */}
      <div className="mb-4 flex flex-1 flex-col gap-2">
        {plan.features.map((f) => (
          <div key={f} className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="flex h-3.5 w-3.5 items-center justify-center rounded-full bg-purple-50">
                <Check size={8} weight="bold" className="text-purple-500" />
              </span>
              <span className="text-[11px] text-zinc-600">{f}</span>
            </div>
            <Info size={12} className="text-zinc-300" />
          </div>
        ))}
      </div>

      {/* CTA */}
      <Button
        variant={plan.buttonVariant}
        size="small"
        onClick={() => onSelect(plan.key)}
        className="w-full"
      >
        {plan.cta}
      </Button>
    </div>
  );
}

/* ─── Main Step Component ─── */

export function SubscriptionStep() {
  const [billing, setBilling] = useState<"monthly" | "yearly">("monthly");
  const [countryIdx, setCountryIdx] = useState(0);
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);

  const country = COUNTRIES[countryIdx];
  const isYearly = billing === "yearly";

  function handlePlanSelect(planKey: string) {
        // Team plan opens the enterprise intake form
    if (planKey === "ENTERPRISE") {
      window.open("https://tally.so/r/2Eb9zj", "_blank");
      return;
    }
    // TODO: integrate with Stripe checkout / backend subscription API
    // For now, advance to the preparing step
    nextStep();
  }

  return (
    <FadeIn>
      <div className="flex w-full flex-col items-center gap-4 px-4">
        {/* Logo */}
        <AutoGPTLogo
          className="relative right-[3rem] h-24 w-[12rem]"
          hideText
        />

        {/* Heading */}
        <div className="flex flex-col items-center gap-1 text-center">
          <Text variant="h3">
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

        {/* Billing toggle – centered */}
        <div className="flex w-full justify-center">
          <div className="inline-flex rounded-full border border-zinc-300 bg-white p-[3px]">
            {(["monthly", "yearly"] as const).map((cycle) => (
              <button
                key={cycle}
                type="button"
                onClick={() => setBilling(cycle)}
                className={cn(
                  "rounded-full border-none px-4 py-1.5 text-xs font-medium transition-all",
                  billing === cycle
                    ? "bg-white text-zinc-900 shadow-sm"
                    : "bg-transparent text-zinc-400",
                )}
              >
                {cycle === "monthly" ? (
                  "Monthly billing"
                ) : (
                  <>
                    Yearly billing{" "}
                    <span className="ml-1.5 text-[11px] font-semibold text-purple-500">
                      Save 15%
                    </span>
                  </>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Plan cards */}
        <div className="mt-2 grid w-full max-w-[960px] grid-cols-3 gap-4">
          {PLANS.map((plan) => (
            <PlanCard
              key={plan.key}
              plan={plan}
              country={country}
              isYearly={isYearly}
              onSelect={handlePlanSelect}
            />
          ))}
        </div>

        {/* Custom link */}
        <Text variant="small" className="!text-zinc-400">
          Need something custom?{" "}
          <a
            href="mailto:sales@agpt.co"
            className="font-medium text-purple-500 no-underline hover:text-purple-600"
          >
            Talk to sales.
          </a>
        </Text>

        {/* Footer: country selector right-aligned */}
        <div className="relative flex w-full max-w-[960px] items-center justify-end">
          <CountrySelector selected={countryIdx} onSelect={setCountryIdx} />
        </div>
      </div>
    </FadeIn>
  );
}

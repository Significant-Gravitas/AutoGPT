interface TierUpgrade {
  /** An upgrade starts a fresh billing month (see `upgrade_resets_billing_date`). */
  resetsBillingDate: boolean;
  /** The new plan's price for one period, formatted; "" when unknown. */
  newPrice: string;
  cycle: "monthly" | "yearly";
}

/** What the upgrade dialog says will be charged and what happens to usage. */
export function tierUpgradeBody({
  resetsBillingDate,
  newPrice,
  cycle,
}: TierUpgrade) {
  const unit = cycle === "yearly" ? "year" : "month";
  if (resetsBillingDate) {
    return [
      newPrice
        ? `You pay the full ${newPrice} today, minus a credit for the unused part of your current plan.`
        : "You pay the new plan’s full price today, minus a credit for the unused part of your current plan.",
      `Your billing date moves to today${newPrice ? `, and your plan renews at ${newPrice} per ${unit}` : ""}.`,
      "Your usage limits start fresh.",
    ].join(" ");
  }
  return [
    `You'll be charged the prorated difference immediately for the rest of your ${cycle} period.`,
    newPrice
      ? `After this period, your plan renews at ${newPrice} per ${unit}.`
      : "",
    "Your current usage carries over into the higher allowance.",
  ]
    .filter(Boolean)
    .join(" ");
}

interface YearlySwitch {
  resetsBillingDate: boolean;
  yearlyPrice: string;
  /** The current period's end, formatted; null when unknown. */
  periodEndLabel: string | null;
}

/** When a plan switched to yearly billing next renews. */
export function yearlyRenewsLine({
  resetsBillingDate,
  yearlyPrice,
  periodEndLabel,
}: YearlySwitch) {
  if (resetsBillingDate)
    return {
      label: "Renews",
      text: `at ${yearlyPrice}/year, a year from today. Your usage limits start fresh.`,
    };
  return periodEndLabel
    ? { label: "Renews", text: `${yearlyPrice}/year on ${periodEndLabel}.` }
    : { label: "Renews", text: `at ${yearlyPrice}/year after this period.` };
}

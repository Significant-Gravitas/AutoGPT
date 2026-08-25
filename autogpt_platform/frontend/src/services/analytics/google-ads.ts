import {
  PLANS,
  YEARLY_PRICE_FACTOR,
} from "@/components/molecules/PlanCard/plans";
import { environment } from "@/services/environment";
import { gtag } from "./gtag";

// One entry per conversion action in the Google Ads account. The action's
// label comes from NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS so the account can
// be rewired without a deploy.
export const ADS_CONVERSIONS = [
  "sign_up",
  "begin_checkout",
  "subscribe",
  "onboarding_complete",
  "top_up",
] as const;
export type AdsConversion = (typeof ADS_CONVERSIONS)[number];

interface ConversionOptions {
  value?: number;
  currency?: string;
  // Google Ads counts one conversion per transaction_id, so a refresh of the
  // page that fired it does not count twice.
  transactionID?: string;
  // Enhanced conversions: gtag hashes the address before it leaves the browser.
  email?: string;
}

export function trackAdsConversion(
  name: AdsConversion,
  options: ConversionOptions = {},
): boolean {
  const adsID = environment.getGoogleAdsID();
  if (!adsID) return false;
  const label = parseConversionLabels(
    environment.getGoogleAdsConversionLabels(),
  )[name];
  if (!label) return false;

  const params: Record<string, unknown> = { send_to: `${adsID}/${label}` };
  if (options.value !== undefined) {
    params.value = options.value;
    params.currency = options.currency ?? "USD";
  }
  if (options.transactionID) params.transaction_id = options.transactionID;
  if (options.email) params.user_data = { email: options.email };

  return gtag("event", "conversion", params);
}

// Client-side navigations don't reload the tag, so the Ads destination gets
// its page views from the router. GA4 is left alone on purpose.
export function trackAdsPageView(path: string): boolean {
  const adsID = environment.getGoogleAdsID();
  if (!adsID) return false;
  return gtag("event", "page_view", { send_to: adsID, page_path: path });
}

// "sign_up=AbCdEf,subscribe=GhIjKl" → { sign_up: "AbCdEf", subscribe: "GhIjKl" }
export function parseConversionLabels(
  raw: string | undefined,
): Partial<Record<AdsConversion, string>> {
  const labels: Partial<Record<AdsConversion, string>> = {};
  for (const part of (raw ?? "").split(",")) {
    const [key, label] = part.split("=").map((piece) => piece.trim());
    if (isAdsConversion(key) && label) labels[key] = label;
  }
  return labels;
}

function isAdsConversion(value: string | undefined): value is AdsConversion {
  return ADS_CONVERSIONS.includes(value as AdsConversion);
}

// USD amount Stripe charges for a plan on the given cycle, for the
// conversion's value. Unknown plans (Team, contact sales) have no value.
export function getSubscriptionValue(
  plan: string | null,
  cycle: string | null,
): number | undefined {
  const monthly = PLANS.find((candidate) => candidate.key === plan)?.usdMonthly;
  if (monthly == null) return undefined;
  const amount =
    cycle === "yearly" ? monthly * YEARLY_PRICE_FACTOR * 12 : monthly;
  return Math.round(amount * 100) / 100;
}

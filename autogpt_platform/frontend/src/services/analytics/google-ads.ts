import {
  PLANS,
  YEARLY_PRICE_FACTOR,
} from "@/components/molecules/PlanCard/plans";
import { consent } from "@/services/consent/cookies";
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
  const label = conversionLabels()[name];
  if (!label) return false;

  const params: Record<string, unknown> = { send_to: `${adsID}/${label}` };
  if (options.value !== undefined) {
    params.value = options.value;
    params.currency = options.currency ?? "USD";
  }
  // Identifiers ride along only on an affirmative yes. The aggregate
  // (cookieless) conversion still counts either way.
  if (mayReportIdentifiers()) {
    if (options.transactionID) params.transaction_id = options.transactionID;
    if (options.email) params.user_data = { email: options.email };
  }

  return gtag("event", "conversion", params);
}

// An unanswered banner is not a yes. Outside the EEA/UK/CH the Consent Mode
// default is `granted`, but that gate lives in the tag's `region` parameter and
// Google resolves it by IP — the browser has no region signal of its own. So
// treating "no answer" as consent would hand Google an email it was told to
// redact for every unanswered visitor in a denied-by-default region, which is
// the vendor dependency this gate exists to remove.
function mayReportIdentifiers(): boolean {
  const preferences = consent.load();
  return preferences.hasConsented && preferences.advertising;
}

// The labels come from a build-time env var; parse once per distinct value so
// a conversion doesn't re-split the string every time.
let cachedLabelsRaw: string | undefined;
let cachedLabels: Partial<Record<AdsConversion, string>> = {};

function conversionLabels(): Partial<Record<AdsConversion, string>> {
  const raw = environment.getGoogleAdsConversionLabels();
  if (raw !== cachedLabelsRaw) {
    cachedLabelsRaw = raw;
    cachedLabels = parseConversionLabels(raw);
  }
  return cachedLabels;
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
  raw: string,
): Partial<Record<AdsConversion, string>> {
  const labels: Partial<Record<AdsConversion, string>> = {};
  for (const part of raw.split(",")) {
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
  const definition = PLANS.find((candidate) => candidate.key === plan);
  const monthly = definition?.usdMonthly;
  if (!definition || monthly == null) return undefined;
  if (cycle !== "yearly") return roundUSD(monthly);
  // usdYearly is the amount Stripe actually charges wherever the surface knows
  // it; the factor is only the fallback the pricing cards compute from
  // (computePricing.ts). Bidding on the displayed number matters.
  return roundUSD(definition.usdYearly ?? monthly * YEARLY_PRICE_FACTOR * 12);
}

// Stripe amounts arrive in cents; Ads wants the major unit.
export function centsToUSD(cents: number | undefined): number | undefined {
  if (cents == null) return undefined;
  return roundUSD(cents / 100);
}

function roundUSD(amount: number): number {
  return Math.round(amount * 100) / 100;
}

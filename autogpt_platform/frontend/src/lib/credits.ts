// The backend meters spend in credits, but the UI only ever talks dollars:
// every credit amount crossing into a view goes through these helpers, and
// every dollar amount typed by a user comes back out as credits.
export const CREDITS_PER_USD = 100;

export function creditsToUsdLabel(credits: number): string {
  const dollars = credits / CREDITS_PER_USD;
  return Number.isInteger(dollars)
    ? `$${dollars.toLocaleString()}`
    : `$${dollars.toFixed(2)}`;
}

export function usdToCredits(dollars: number): number {
  return Math.round(dollars * CREDITS_PER_USD);
}

// Accepts what a person would type into a dollar field — "5", "$5", "5.25",
// "1,000" — and rejects anything finer than cents. Thousands separators are
// validated in place rather than stripped first, so "1,2" stays invalid
// instead of silently becoming $12. Returns credits so callers never have to
// remember the conversion.
const USD_INPUT = /^\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?$/;

export function parseUsdToCredits(
  value: string,
  maxCredits: number,
): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (!USD_INPUT.test(trimmed)) return null;
  const credits = usdToCredits(Number(trimmed.replace(/[$,]/g, "")));
  if (credits < 0 || credits > maxCredits) return null;
  return credits;
}

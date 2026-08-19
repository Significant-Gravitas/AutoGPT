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

// Accepts what a person would type into a dollar field — "5", "$5", "5.25" —
// and rejects anything finer than cents. Returns credits so callers never
// have to remember the conversion.
export function parseUsdToCredits(
  value: string,
  maxCredits: number,
): number | null {
  const trimmed = value.trim().replace(/^\$/, "").replace(/,/g, "");
  if (!trimmed) return null;
  if (!/^\d+(\.\d{1,2})?$/.test(trimmed)) return null;
  const credits = usdToCredits(Number(trimmed));
  if (credits < 0 || credits > maxCredits) return null;
  return credits;
}

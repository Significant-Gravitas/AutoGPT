// total_spent arrives as integer credits (cents), matching useCredits'
// formatCredits — 100 == $1.00.
export function formatSpend(cents: number): string {
  const value = Math.abs(cents);
  const sign = cents < 0 ? "-" : "";
  return `${sign}$${(value / 100).toFixed(2)}`;
}

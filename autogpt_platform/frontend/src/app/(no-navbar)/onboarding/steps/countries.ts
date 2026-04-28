/**
 * Stripe-supported billing countries with exchange rates.
 * Rates sourced from x-rates.com (Apr 28, 2026 09:03 UTC).
 */

export interface Country {
  name: string;
  flag: string;
  code: string;
  symbol: string;
  rate: number;
}

export const COUNTRIES: Country[] = [
  { name: "United States", flag: "🇺🇸", code: "USD", symbol: "$", rate: 1 },
  { name: "United Kingdom", flag: "🇬🇧", code: "GBP", symbol: "£", rate: 0.7406 },
  { name: "European Union", flag: "🇪🇺", code: "EUR", symbol: "€", rate: 0.8547 },
  { name: "Canada", flag: "🇨🇦", code: "CAD", symbol: "CA$", rate: 1.3649 },
  { name: "Australia", flag: "🇦🇺", code: "AUD", symbol: "A$", rate: 1.3938 },
  { name: "Japan", flag: "🇯🇵", code: "JPY", symbol: "¥", rate: 159.52 },
  { name: "Singapore", flag: "🇸🇬", code: "SGD", symbol: "S$", rate: 1.2763 },
  { name: "Switzerland", flag: "🇨🇭", code: "CHF", symbol: "CHF ", rate: 0.7889 },
  { name: "New Zealand", flag: "🇳🇿", code: "NZD", symbol: "NZ$", rate: 1.698 },
  { name: "Brazil", flag: "🇧🇷", code: "BRL", symbol: "R$", rate: 4.9889 },
  { name: "Mexico", flag: "🇲🇽", code: "MXN", symbol: "MX$", rate: 17.4206 },
  { name: "India", flag: "🇮🇳", code: "INR", symbol: "₹", rate: 94.5316 },
  { name: "South Korea", flag: "🇰🇷", code: "KRW", symbol: "₩", rate: 1473.35 },
  { name: "Denmark", flag: "🇩🇰", code: "DKK", symbol: "kr ", rate: 6.3866 },
  { name: "Norway", flag: "🇳🇴", code: "NOK", symbol: "kr ", rate: 9.2951 },
  { name: "Sweden", flag: "🇸🇪", code: "SEK", symbol: "kr ", rate: 9.255 },
  { name: "Poland", flag: "🇵🇱", code: "PLN", symbol: "zł", rate: 3.6298 },
  { name: "Czech Republic", flag: "🇨🇿", code: "CZK", symbol: "Kč", rate: 20.829 },
  { name: "Romania", flag: "🇷🇴", code: "RON", symbol: "lei ", rate: 4.353 },
  { name: "Hungary", flag: "🇭🇺", code: "HUF", symbol: "Ft ", rate: 311.94 },
  { name: "Hong Kong", flag: "🇭🇰", code: "HKD", symbol: "HK$", rate: 7.836 },
  { name: "Thailand", flag: "🇹🇭", code: "THB", symbol: "฿", rate: 32.487 },
  { name: "Malaysia", flag: "🇲🇾", code: "MYR", symbol: "RM ", rate: 3.951 },
  { name: "Israel", flag: "🇮🇱", code: "ILS", symbol: "₪", rate: 2.9848 },
  { name: "South Africa", flag: "🇿🇦", code: "ZAR", symbol: "R ", rate: 16.5825 },
  { name: "UAE", flag: "🇦🇪", code: "AED", symbol: "AED ", rate: 3.6725 },
  { name: "Turkey", flag: "🇹🇷", code: "TRY", symbol: "₺", rate: 45.049 },
  { name: "Philippines", flag: "🇵🇭", code: "PHP", symbol: "₱", rate: 61.286 },
];

const ZERO_DECIMAL_CODES = new Set(["JPY", "KRW", "HUF", "CLP"]);

/**
 * Format a price value for display with the correct symbol and decimal handling.
 * Zero-decimal currencies (JPY, KRW, HUF, CLP) show no decimals.
 * Values >= 100 are rounded to integers for cleaner display.
 */
export function formatPrice(
  value: number,
  code: string,
  symbol: string,
): string {
  if (ZERO_DECIMAL_CODES.has(code)) {
    return symbol + Math.round(value).toLocaleString("en-US");
  }
  if (value >= 100) {
    return symbol + Math.round(value).toLocaleString("en-US");
  }
  return symbol + value.toFixed(2);
}

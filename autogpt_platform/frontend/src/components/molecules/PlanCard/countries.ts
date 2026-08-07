/**
 * Stripe-supported billing countries with exchange rates.
 * Rates sourced from x-rates.com (Apr 28, 2026 09:03 UTC).
 */

export interface Country {
  name: string;
  flag: string;
  // ISO 3166-1 alpha-2 (or "EU" for the European Union grouping). This is the
  // identity field — selection and persistence are keyed by countryCode.
  countryCode: string;
  // ISO 4217 currency. Used for pricing math, formatting, and Stripe.
  currencyCode: string;
  symbol: string;
  rate: number;
}

export const COUNTRIES: Country[] = [
  {
    name: "United States",
    flag: "🇺🇸",
    countryCode: "US",
    currencyCode: "USD",
    symbol: "$",
    rate: 1,
  },
  {
    name: "United Kingdom",
    flag: "🇬🇧",
    countryCode: "GB",
    currencyCode: "GBP",
    symbol: "£",
    rate: 0.7406,
  },
  {
    name: "European Union",
    flag: "🇪🇺",
    countryCode: "EU",
    currencyCode: "EUR",
    symbol: "€",
    rate: 0.8547,
  },
  {
    name: "Canada",
    flag: "🇨🇦",
    countryCode: "CA",
    currencyCode: "CAD",
    symbol: "CA$",
    rate: 1.3649,
  },
  {
    name: "Australia",
    flag: "🇦🇺",
    countryCode: "AU",
    currencyCode: "AUD",
    symbol: "A$",
    rate: 1.3938,
  },
  {
    name: "Japan",
    flag: "🇯🇵",
    countryCode: "JP",
    currencyCode: "JPY",
    symbol: "¥",
    rate: 159.52,
  },
  {
    name: "Singapore",
    flag: "🇸🇬",
    countryCode: "SG",
    currencyCode: "SGD",
    symbol: "S$",
    rate: 1.2763,
  },
  {
    name: "Switzerland",
    flag: "🇨🇭",
    countryCode: "CH",
    currencyCode: "CHF",
    symbol: "CHF ",
    rate: 0.7889,
  },
  {
    name: "New Zealand",
    flag: "🇳🇿",
    countryCode: "NZ",
    currencyCode: "NZD",
    symbol: "NZ$",
    rate: 1.698,
  },
  {
    name: "Brazil",
    flag: "🇧🇷",
    countryCode: "BR",
    currencyCode: "BRL",
    symbol: "R$",
    rate: 4.9889,
  },
  {
    name: "Mexico",
    flag: "🇲🇽",
    countryCode: "MX",
    currencyCode: "MXN",
    symbol: "MX$",
    rate: 17.4206,
  },
  {
    name: "India",
    flag: "🇮🇳",
    countryCode: "IN",
    currencyCode: "INR",
    symbol: "₹",
    rate: 94.5316,
  },
  {
    name: "South Korea",
    flag: "🇰🇷",
    countryCode: "KR",
    currencyCode: "KRW",
    symbol: "₩",
    rate: 1473.35,
  },
  {
    name: "Denmark",
    flag: "🇩🇰",
    countryCode: "DK",
    currencyCode: "DKK",
    symbol: "kr ",
    rate: 6.3866,
  },
  {
    name: "Norway",
    flag: "🇳🇴",
    countryCode: "NO",
    currencyCode: "NOK",
    symbol: "kr ",
    rate: 9.2951,
  },
  {
    name: "Sweden",
    flag: "🇸🇪",
    countryCode: "SE",
    currencyCode: "SEK",
    symbol: "kr ",
    rate: 9.255,
  },
  {
    name: "Poland",
    flag: "🇵🇱",
    countryCode: "PL",
    currencyCode: "PLN",
    symbol: "zł",
    rate: 3.6298,
  },
  {
    name: "Czech Republic",
    flag: "🇨🇿",
    countryCode: "CZ",
    currencyCode: "CZK",
    symbol: "Kč",
    rate: 20.829,
  },
  {
    name: "Romania",
    flag: "🇷🇴",
    countryCode: "RO",
    currencyCode: "RON",
    symbol: "lei ",
    rate: 4.353,
  },
  {
    name: "Hungary",
    flag: "🇭🇺",
    countryCode: "HU",
    currencyCode: "HUF",
    symbol: "Ft ",
    rate: 311.94,
  },
  {
    name: "Hong Kong",
    flag: "🇭🇰",
    countryCode: "HK",
    currencyCode: "HKD",
    symbol: "HK$",
    rate: 7.836,
  },
  {
    name: "Thailand",
    flag: "🇹🇭",
    countryCode: "TH",
    currencyCode: "THB",
    symbol: "฿",
    rate: 32.487,
  },
  {
    name: "Malaysia",
    flag: "🇲🇾",
    countryCode: "MY",
    currencyCode: "MYR",
    symbol: "RM ",
    rate: 3.951,
  },
  {
    name: "Israel",
    flag: "🇮🇱",
    countryCode: "IL",
    currencyCode: "ILS",
    symbol: "₪",
    rate: 2.9848,
  },
  {
    name: "South Africa",
    flag: "🇿🇦",
    countryCode: "ZA",
    currencyCode: "ZAR",
    symbol: "R ",
    rate: 16.5825,
  },
  {
    name: "UAE",
    flag: "🇦🇪",
    countryCode: "AE",
    currencyCode: "AED",
    symbol: "AED ",
    rate: 3.6725,
  },
  {
    name: "Turkey",
    flag: "🇹🇷",
    countryCode: "TR",
    currencyCode: "TRY",
    symbol: "₺",
    rate: 45.049,
  },
  {
    name: "Philippines",
    flag: "🇵🇭",
    countryCode: "PH",
    currencyCode: "PHP",
    symbol: "₱",
    rate: 61.286,
  },
];

const ZERO_DECIMAL_CODES = new Set(["JPY", "KRW", "HUF", "CLP"]);

// Zero-decimal currencies show no decimals (Stripe convention).
// All other currencies always show two decimals so cents aren't dropped on
// large amounts (e.g. yearly Max in BRL).
export function formatPrice(
  value: number,
  currencyCode: string,
  symbol: string,
): string {
  if (ZERO_DECIMAL_CODES.has(currencyCode)) {
    return symbol + Math.round(value).toLocaleString("en-US");
  }
  return (
    symbol +
    value.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })
  );
}

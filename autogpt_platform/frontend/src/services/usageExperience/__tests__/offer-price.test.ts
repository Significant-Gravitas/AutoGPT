import { expect, it } from "vitest";
import { formatOfferPrice } from "../helpers";

it.each([
  [5000, "USD", "$50"],
  [5050, "USD", "$50.50"],
  [5000, "JPY", "¥5,000"],
  [50123, "BHD", "BHD 50.123"],
] as const)(
  "keeps accepted %s %s offers consistent with their currency's minor units",
  (amount, currency, expected) => {
    expect(formatOfferPrice(amount, currency)?.replace(/\s+/g, " ")).toBe(
      expected,
    );
  },
);

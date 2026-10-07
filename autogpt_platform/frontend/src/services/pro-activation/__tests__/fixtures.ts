import type { ActivationResponse } from "@/app/api/__generated__/models/activationResponse";

export const quote: ActivationResponse = {
  id: "attempt-1",
  status: "confirmation_required",
  terms_token: "a".repeat(64),
  return_to: "/copilot/thread?resume=1#draft",
  terms: {
    plan: "PRO",
    price_id: "accepted-price",
    accepted_offer_token: "accepted-offer",
    amount_due: 6123,
    currency: "gbp",
    billing_interval: "year",
    billing_interval_count: 1,
    renewal_unit_amount: 8500,
    renewal_terms: "Renews annually. Cancel before renewal.",
    renewal_discounts: [{ percent_off: 20, duration: "once" }],
    renewal_tax: {
      automatic: true,
      price_tax_behavior: "exclusive",
      rates: [
        {
          display_name: "VAT",
          percentage: 20,
          inclusive: false,
          country: "GB",
        },
      ],
    },
    expires_at: new Date("2099-01-01T00:00:00Z"),
  },
};

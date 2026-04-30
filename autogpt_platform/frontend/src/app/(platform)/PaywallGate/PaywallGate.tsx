"use client";

import { ReactNode } from "react";
import { usePathname } from "next/navigation";
import { useGetSubscriptionStatus } from "@/app/api/__generated__/endpoints/credits/credits";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { PaywallModal } from "./PaywallModal";

// Routes that bypass the paywall regardless of subscription state — primarily
// the credits page itself (the modal would render on top of itself), auth
// flows, account management, and admin areas the user needs even when locked
// out of AutoPilot.
const PAYWALL_EXEMPT_PREFIXES = [
  "/profile",
  "/admin",
  "/auth",
  "/login",
  "/signup",
  "/reset-password",
  "/error",
  "/unauthorized",
  "/health",
];

export function PaywallGate({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const { data: subscription, isLoading } = useGetSubscriptionStatus({
    query: {
      select: (res) => (res.status === 200 ? res.data : null),
      // Skip the call entirely when the flag is off — beta cohort never hits
      // the paywall, no need to read /credits/subscription on every page load.
      enabled: isPaymentEnabled === true,
    },
  });

  const isExempt = PAYWALL_EXEMPT_PREFIXES.some((p) => pathname.startsWith(p));

  // Gate on the DB tier rather than has_active_stripe_subscription so a
  // transient Stripe outage (which would set the latter to False even for
  // active subscribers) doesn't lock out paying users. The DB tier is set
  // by Stripe webhooks and persists; only fresh users without any tier
  // change land on NO_TIER (the explicit "no active subscription" state).
  const shouldGate =
    !!isPaymentEnabled &&
    !isLoading &&
    !isExempt &&
    !!subscription &&
    subscription.tier === "NO_TIER";

  return (
    <>
      {children}
      {shouldGate && <PaywallModal />}
    </>
  );
}

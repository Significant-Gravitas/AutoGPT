"use client";

import { ReactNode, useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useGetSubscriptionStatus } from "@/app/api/__generated__/endpoints/credits/credits";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

const PAYWALL_PATH = "/profile/credits";

// Routes that bypass the paywall regardless of subscription state — primarily
// the credits page itself (otherwise we redirect-loop), auth flows, account
// management, and admin areas the user needs even when locked out of AutoPilot.
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
  const router = useRouter();
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

  useEffect(() => {
    if (!isPaymentEnabled || isLoading || isExempt) return;
    if (!subscription) return;
    if (subscription.has_active_stripe_subscription) return;
    router.replace(PAYWALL_PATH);
  }, [isPaymentEnabled, isLoading, isExempt, subscription, router]);

  return <>{children}</>;
}

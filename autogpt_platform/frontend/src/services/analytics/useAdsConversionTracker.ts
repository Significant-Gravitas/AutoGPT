import { useAuth } from "@/lib/auth/hooks/useAuth";
import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";
import { consumeAccountCreatedFlag } from "./account-created-cookie";
import {
  getSubscriptionValue,
  trackAdsConversion,
  trackAdsPageView,
} from "./google-ads";

export function useAdsConversionTracker() {
  const { user, isUserLoading } = useAuth();
  const pathname = usePathname();
  const checkoutReportedRef = useRef(false);
  const lastTrackedPathRef = useRef<string | null>(null);

  // Waits for the session so the user id (dedup) and email (enhanced
  // conversions) can ride along. Re-runs on navigation because the email
  // signup sets the flag and then moves on client-side, without a reload.
  useEffect(() => {
    if (isUserLoading) return;

    if (!checkoutReportedRef.current) {
      checkoutReportedRef.current = true;
      // Stripe sends the user back with a full page load, so the checkout
      // result is only ever in the URL at mount time.
      trackCheckoutReturn(
        new URLSearchParams(window.location.search),
        user?.email,
      );
    }

    const method = consumeAccountCreatedFlag();
    if (!method) return;
    trackAdsConversion("sign_up", {
      transactionID: user?.id,
      email: user?.email,
    });
  }, [user?.id, isUserLoading, pathname]);

  // The tag's own config call already reports the first page; only
  // client-side navigations need a page_view from here.
  useEffect(() => {
    if (lastTrackedPathRef.current === null) {
      lastTrackedPathRef.current = pathname;
      return;
    }
    if (pathname === lastTrackedPathRef.current) return;
    lastTrackedPathRef.current = pathname;
    trackAdsPageView(pathname);
  }, [pathname]);
}

function trackCheckoutReturn(params: URLSearchParams, email?: string) {
  const sessionID = params.get("session_id") ?? undefined;

  if (params.get("subscription") === "success") {
    trackAdsConversion("subscribe", {
      value: getSubscriptionValue(params.get("plan"), params.get("cycle")),
      transactionID: sessionID,
      email,
    });
  }

  if (params.get("topup") === "success") {
    trackAdsConversion("top_up", { transactionID: sessionID, email });
  }
}

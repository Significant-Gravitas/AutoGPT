import { useAuth } from "@/lib/auth/hooks/useAuth";
import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";
import {
  clearAccountCreatedFlag,
  readAccountCreatedFlag,
} from "./account-created-cookie";
import {
  getSubscriptionValue,
  trackAdsConversion,
  trackAdsPageView,
} from "./google-ads";

export function useAdsConversionTracker() {
  const { user, isUserLoading } = useAuth();
  const pathname = usePathname();
  const checkoutReportedRef = useRef(false);
  const checkoutParamsRef = useRef<URLSearchParams | null>(null);
  const lastTrackedPathRef = useRef<string | null>(null);

  // Waits for the session itself, not just for loading to finish: on the
  // post-signup client navigation the effect runs once with `user` still
  // undefined, and a conversion sent then carries no id to dedup on and no
  // email for enhanced conversions. Nothing here is reported without a user.
  // Re-runs on navigation because the email signup sets the flag and then
  // moves on client-side, without a reload.
  useEffect(() => {
    // Snapshot before the session guard. Stripe's result is only in the URL on
    // the landing render, and a client-side navigation — including the credits
    // page stripping its own query — can drop it before a retry runs, which
    // would latch the ref on an empty query and lose the conversion for good.
    const checkoutParams =
      checkoutParamsRef.current ?? new URLSearchParams(window.location.search);
    checkoutParamsRef.current = checkoutParams;

    if (isUserLoading || !user) return;

    if (!checkoutReportedRef.current) {
      // Latch only once it actually reached the tag — the tag loads
      // afterInteractive and may not exist yet on this pass.
      checkoutReportedRef.current = trackCheckoutReturn(
        checkoutParams,
        user.email,
      );
    }

    const method = readAccountCreatedFlag();
    if (!method) return;
    const reported = trackAdsConversion("sign_up", {
      transactionID: user.id,
      email: user.email,
    });
    // The flag is the only record that a signup happened; keep it until the
    // conversion is really out, so the next navigation can retry.
    if (reported) clearAccountCreatedFlag();
  }, [user, isUserLoading, pathname]);

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

// Returns whether the checkout result has been dealt with: true when there was
// nothing to report, or when every conversion reached the tag.
function trackCheckoutReturn(params: URLSearchParams, email?: string): boolean {
  const sessionID = params.get("session_id") ?? undefined;
  let handled = true;

  if (params.get("subscription") === "success") {
    handled =
      trackAdsConversion("subscribe", {
        value: getSubscriptionValue(params.get("plan"), params.get("cycle")),
        transactionID: sessionID,
        email,
      }) && handled;
  }

  if (params.get("topup") === "success") {
    handled =
      trackAdsConversion("top_up", { transactionID: sessionID, email }) &&
      handled;
  }

  return handled;
}

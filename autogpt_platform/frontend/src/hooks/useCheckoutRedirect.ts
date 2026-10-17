import { useEffect, useState } from "react";
import { useMountEffect } from "./useMountEffect";

// Long enough for Stripe to load on a slow connection, short enough that a
// navigation that never happens (blocked, or the user pressed Stop) doesn't
// leave the button busy for good.
const CHECKOUT_REDIRECT_TIMEOUT_MS = 15_000;

// Keeps a checkout button busy from the moment it hands off to Stripe until
// the page actually leaves, so a second click can't start a second checkout.
// Back from Stripe restores this page from the back/forward cache with that
// busy state intact, so it is released again on `pageshow`.
export function useCheckoutRedirect() {
  const [isRedirecting, setIsRedirecting] = useState(false);

  useMountEffect(() => {
    function onPageShow(event: PageTransitionEvent) {
      if (event.persisted) setIsRedirecting(false);
    }
    window.addEventListener("pageshow", onPageShow);
    return () => window.removeEventListener("pageshow", onPageShow);
  });

  useEffect(() => {
    if (!isRedirecting) return;
    const timeout = setTimeout(
      () => setIsRedirecting(false),
      CHECKOUT_REDIRECT_TIMEOUT_MS,
    );
    return () => clearTimeout(timeout);
  }, [isRedirecting]);

  // Call right after starting the navigation, so a navigation that throws
  // never marks the button busy.
  function markRedirecting() {
    setIsRedirecting(true);
  }

  return { isRedirecting, markRedirecting };
}

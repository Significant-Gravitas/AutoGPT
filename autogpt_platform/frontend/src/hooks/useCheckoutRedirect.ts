import { useState, useSyncExternalStore } from "react";

// No browser event says a navigation has committed, so a click after this
// long on a page that still hasn't left is the residual window we accept.
const CHECKOUT_REDIRECT_TIMEOUT_MS = 20_000;

// One redirect per page, shared by every checkout button on it: a plan, trial
// or top-up button that hands off to Stripe makes all of them busy until the
// page leaves. The id tells the button that started it from the others.
let redirectID: number | null = null;
let lastRedirectID = 0;
let timeout: ReturnType<typeof setTimeout> | undefined;
const subscribers = new Set<() => void>();

function notify() {
  subscribers.forEach((subscriber) => subscriber());
}

function subscribe(subscriber: () => void) {
  subscribers.add(subscriber);
  return () => {
    subscribers.delete(subscriber);
  };
}

function getRedirectID() {
  return redirectID;
}

function getServerRedirectID() {
  return null;
}

// Back from Stripe restores this page from the back/forward cache with the
// busy state intact, so a restore releases it.
function onPageShow(event: PageTransitionEvent) {
  if (event.persisted) resetCheckoutRedirect();
}

function startRedirect() {
  if (redirectID === null) window.addEventListener("pageshow", onPageShow);
  clearTimeout(timeout);
  timeout = setTimeout(resetCheckoutRedirect, CHECKOUT_REDIRECT_TIMEOUT_MS);
  lastRedirectID += 1;
  redirectID = lastRedirectID;
  notify();
  return redirectID;
}

export function resetCheckoutRedirect() {
  if (redirectID === null) return;
  window.removeEventListener("pageshow", onPageShow);
  clearTimeout(timeout);
  timeout = undefined;
  redirectID = null;
  notify();
}

export function useCheckoutRedirect() {
  const currentID = useSyncExternalStore(
    subscribe,
    getRedirectID,
    getServerRedirectID,
  );
  const [ownID, setOwnID] = useState<number | null>(null);

  // Call right after starting the navigation, so a navigation that throws
  // never marks the page busy.
  function markRedirecting() {
    setOwnID(startRedirect());
  }

  return {
    isRedirecting: currentID !== null,
    isRedirectingHere: currentID !== null && currentID === ownID,
    markRedirecting,
  };
}

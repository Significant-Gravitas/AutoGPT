import { analytics } from "@/services/analytics";

const PAYWALL_VIEW_SESSION_KEY = "paywall_view_tracked";

/**
 * Reports that the user was actually shown the paywall.
 *
 * Onboarding's steps all share the `/onboarding` URL (the `?step=` param is
 * stripped by DataFast), so without this event the funnel can't tell "never
 * reached the subscription screen" from "saw the price and declined" — which
 * is the only question that matters between signup and payment.
 *
 * Deliberately untagged by pricing arm. `posthog.init` runs in an effect with
 * no bootstrap, so the experiment variant is still unresolved when this step
 * mounts and would record "control" for everyone; PostHog records arm
 * assignment itself, so segmentation belongs there rather than here.
 *
 * Fires at most once per tab. Two paths would otherwise double-count:
 * cancelling Stripe checkout returns to `?step=1&subscription=cancelled` as a
 * full navigation that remounts this step, and a render committed before the
 * wizard's init effect corrects `currentStep` can briefly mount the paywall at
 * the store default of step 1. A genuinely new view (new tab or session) finds
 * the key unset and reports normally.
 */
export function trackPaywallView() {
  try {
    if (sessionStorage.getItem(PAYWALL_VIEW_SESSION_KEY)) return;
    sessionStorage.setItem(PAYWALL_VIEW_SESSION_KEY, "1");
  } catch {
    // In-app browsers may block sessionStorage — double-counting a view beats
    // dropping it, so fall through to report either way.
  }

  try {
    analytics.sendDatafastEvent("paywall_view", {});
  } catch {
    // sendDatafastEvent hands the event to the third-party DataFast script,
    // which it calls without a guard of its own. This runs in a mount effect on
    // the screen users pay from, so an exception here would unmount the paywall
    // into an error boundary. Analytics must never be able to take checkout down.
  }
}

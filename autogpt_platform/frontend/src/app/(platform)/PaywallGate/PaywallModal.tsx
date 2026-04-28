"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { SubscriptionTierSection } from "@/app/(platform)/profile/(user)/credits/components/SubscriptionTierSection/SubscriptionTierSection";

// Non-dismissable Stripe paywall. The user can only exit by picking a tier
// (Stripe Checkout or modify-in-place). Reuses the existing tier picker so
// the underlying Checkout / modify wiring stays in one place — the
// SubscriptionTierSection already renders its own "Pick a plan" banner +
// description when tier === "BASIC", so the dialog itself is title-less.
export function PaywallModal() {
  return (
    <Dialog forceOpen controlled={{ isOpen: true, set: () => {} }}>
      <Dialog.Content>
        <SubscriptionTierSection />
      </Dialog.Content>
    </Dialog>
  );
}

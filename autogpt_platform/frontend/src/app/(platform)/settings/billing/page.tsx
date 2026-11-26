"use client";

import { useEffect, useRef, useState } from "react";
import { notFound, useRouter, useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";

import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { UsageCreditsDialog } from "@/components/organisms/UsageCreditsDialog/UsageCreditsDialog";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { useBillingStatus } from "@/hooks/useIsBillingEnabled";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { toast } from "@/components/molecules/Toast/use-toast";
import { getGetSubscriptionStatusQueryKey } from "@/app/api/__generated__/endpoints/credits/credits";
import { getGetTrialsGetTrialStatusQueryKey } from "@/app/api/__generated__/endpoints/trials/trials";

import {
  trackCheckoutAbandoned,
  trackTrialCheckoutAbandoned,
} from "@/services/analytics/monetization-analytics";

import { AutomationCreditsTab } from "./components/AutomationCreditsTab/AutomationCreditsTab";
import { CheckoutAbandonedPrompt } from "./components/CheckoutAbandonedPrompt/CheckoutAbandonedPrompt";
import {
  readAbandonedPlan,
  type AbandonedPlan,
} from "./components/CheckoutAbandonedPrompt/helpers";
import { SubscriptionTab } from "./components/SubscriptionTab/SubscriptionTab";
import { billingTabURL, isBillingTab, type BillingTab } from "./helpers";
import { useTopupCheckoutReturn } from "./useTopupCheckoutReturn";

function resolveInitialTab(searchParams: URLSearchParams): BillingTab {
  const tabParam = searchParams.get("tab");
  if (tabParam && isBillingTab(tabParam)) return tabParam;
  // Topup flow only originates from the Automation Credits tab — when Stripe
  // redirects back, return the user to where they started instead of the
  // Subscription default.
  if (searchParams.get("topup")) return "automation-credits";
  return "subscription";
}

export default function SettingsBillingRoute() {
  const { isBillingEnabled, isResolved } = useBillingStatus();
  if (!isResolved) return null;
  if (!isBillingEnabled) notFound();
  return <SettingsBillingPage />;
}

function SettingsBillingPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const topupStatus = searchParams.get("topup");
  const subscriptionStatus = searchParams.get("subscription");
  const trialStatus = searchParams.get("trial");
  const handledSubscriptionRef = useRef<string | null>(null);
  // Set on return from an upgrade checkout left unpaid; the plan is null when
  // the cancel URL did not name one.
  const [abandonedCheckout, setAbandonedCheckout] = useState<{
    plan: AbandonedPlan | null;
  } | null>(null);
  const [isExplainerOpen, setIsExplainerOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<BillingTab>(() =>
    resolveInitialTab(searchParams),
  );
  // Read once: the redirect handler below drops the query string.
  const [isPlanCheckoutReturn] = useState(
    () => subscriptionStatus === "success",
  );

  function handleTabChange(value: string) {
    if (!isBillingTab(value)) return;
    setActiveTab(value);
    router.replace(billingTabURL(value, searchParams), { scroll: false });
  }

  useEffect(function setBillingDocumentTitle() {
    document.title = "Billing – AutoGPT Platform";
  }, []);

  useTopupCheckoutReturn({ topupStatus, activeTab });

  useEffect(
    function handleSubscriptionRedirect() {
      if (!subscriptionStatus) return;
      if (handledSubscriptionRef.current === subscriptionStatus) return;
      handledSubscriptionRef.current = subscriptionStatus;

      if (subscriptionStatus === "success") {
        toast({
          title: "Confirming your subscription",
          description:
            "Your new plan is being applied. It may take a moment to reflect.",
        });
        queryClient.invalidateQueries({
          queryKey: getGetSubscriptionStatusQueryKey(),
        });
        queryClient.invalidateQueries({
          queryKey: getGetTrialsGetTrialStatusQueryKey(),
        });
      } else if (subscriptionStatus === "cancelled") {
        trackCheckoutAbandoned({
          checkout_kind: "subscription",
          surface: "billing",
        });
        setAbandonedCheckout({ plan: readAbandonedPlan(searchParams) });
      }

      router.replace("/settings/billing");
    },
    [subscriptionStatus, queryClient, router, searchParams],
  );

  useEffect(
    function reportTrialCheckoutCancelled() {
      if (trialStatus === "cancelled") trackTrialCheckoutAbandoned("billing");
    },
    [trialStatus],
  );

  return (
    <div className="mx-auto flex w-full max-w-[900px] flex-col gap-5 px-4 pb-8 sm:px-6">
      <header className="space-y-1">
        <div className="flex items-center gap-1">
          <Text variant="h3" tracking="tight" as="h1">
            Billing &amp; usage
          </Text>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            shape="soft"
            leadingIcon={InformationCircleIcon}
            aria-label="How usage and credits work"
            tone="muted"
            onClick={() => setIsExplainerOpen(true)}
          />
        </div>
        <Text variant="body" tone="secondary">
          Everything you need to manage your plan.
        </Text>
      </header>

      <TabsLine value={activeTab} onValueChange={handleTabChange}>
        <TabsLineList flush className="overflow-x-auto">
          <TabsLineTrigger value="subscription">Subscription</TabsLineTrigger>
          <TabsLineTrigger value="automation-credits">
            Automation Credits
          </TabsLineTrigger>
        </TabsLineList>

        <TabsLineContent value="subscription">
          <SubscriptionTab isPlanCheckoutReturn={isPlanCheckoutReturn} />
        </TabsLineContent>

        <TabsLineContent value="automation-credits">
          <AutomationCreditsTab />
        </TabsLineContent>
      </TabsLine>

      <UsageCreditsDialog
        isOpen={isExplainerOpen}
        onOpenChange={setIsExplainerOpen}
        showBillingLink={false}
      />
      {abandonedCheckout ? (
        <CheckoutAbandonedPrompt plan={abandonedCheckout.plan} />
      ) : null}
    </div>
  );
}

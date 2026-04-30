"use client";

import { useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";

import { Text } from "@/components/atoms/Text/Text";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  getGetSubscriptionStatusQueryKey,
  usePatchV1FulfillCheckoutSession,
} from "@/app/api/__generated__/endpoints/credits/credits";

import { AutomationCreditsTab } from "./components/AutomationCreditsTab/AutomationCreditsTab";
import { SubscriptionTab } from "./components/SubscriptionTab/SubscriptionTab";

export default function SettingsBillingPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const topupStatus = searchParams.get("topup");
  const subscriptionStatus = searchParams.get("subscription");
  const { mutateAsync: fulfillCheckout } = usePatchV1FulfillCheckoutSession();
  const handledTopupRef = useRef<string | null>(null);
  const handledSubscriptionRef = useRef<string | null>(null);

  useEffect(function setBillingDocumentTitle() {
    document.title = "Billing – AutoGPT Platform";
  }, []);

  useEffect(
    function handleTopupRedirect() {
      if (!topupStatus) return;
      // Stripe re-renders the page after redirect — guard so we don't
      // fire fulfillCheckout twice or stack duplicate toasts.
      if (handledTopupRef.current === topupStatus) return;
      handledTopupRef.current = topupStatus;

      if (topupStatus === "success") {
        toast({
          title: "Payment successful",
          description:
            "Your credits will appear shortly. Refresh if you don't see them.",
        });
        fulfillCheckout().catch(() => {
          // Background fulfillment call — webhook is the source of truth,
          // so a failure here is non-blocking.
        });
      } else if (topupStatus === "cancel") {
        toast({
          title: "Payment cancelled",
          description: "Your payment method was not charged.",
          variant: "destructive",
        });
      }

      router.replace("/settings/billing");
    },
    [topupStatus, fulfillCheckout, router],
  );

  useEffect(
    function handleSubscriptionRedirect() {
      if (!subscriptionStatus) return;
      if (handledSubscriptionRef.current === subscriptionStatus) return;
      handledSubscriptionRef.current = subscriptionStatus;

      if (subscriptionStatus === "success") {
        toast({
          title: "Subscription updated",
          description:
            "Your new plan is being applied. It may take a moment to reflect.",
        });
        queryClient.invalidateQueries({
          queryKey: getGetSubscriptionStatusQueryKey(),
        });
      } else if (subscriptionStatus === "cancelled") {
        toast({
          title: "Checkout cancelled",
          description: "Your plan was not changed.",
          variant: "destructive",
        });
      }

      router.replace("/settings/billing");
    },
    [subscriptionStatus, queryClient, router],
  );

  return (
    <div className="flex flex-col gap-6 pb-8">
      <Text variant="h4" as="h1" className="pl-4 leading-[28px] text-textBlack">
        Billing
      </Text>

      <TabsLine defaultValue="subscription">
        <TabsLineList flush className="ml-4">
          <TabsLineTrigger value="subscription">Subscription</TabsLineTrigger>
          <TabsLineTrigger value="automation-credits">
            Automation Credits
          </TabsLineTrigger>
        </TabsLineList>

        <TabsLineContent value="subscription">
          <SubscriptionTab />
        </TabsLineContent>

        <TabsLineContent value="automation-credits">
          <AutomationCreditsTab />
        </TabsLineContent>
      </TabsLine>
    </div>
  );
}

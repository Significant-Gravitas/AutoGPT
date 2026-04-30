"use client";

import { useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { Text } from "@/components/atoms/Text/Text";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { toast } from "@/components/molecules/Toast/use-toast";
import { usePatchV1FulfillCheckoutSession } from "@/app/api/__generated__/endpoints/credits/credits";

import { AutomationCreditsTab } from "./components/AutomationCreditsTab/AutomationCreditsTab";
import { SubscriptionTab } from "./components/SubscriptionTab/SubscriptionTab";

export default function SettingsBillingPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const topupStatus = searchParams.get("topup");
  const { mutateAsync: fulfillCheckout } = usePatchV1FulfillCheckoutSession();
  const handledTopupRef = useRef<string | null>(null);

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

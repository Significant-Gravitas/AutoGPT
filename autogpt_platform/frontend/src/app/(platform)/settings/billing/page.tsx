"use client";

import { useEffect } from "react";

import { Text } from "@/components/atoms/Text/Text";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";

import { AutomationCreditsTab } from "./components/AutomationCreditsTab/AutomationCreditsTab";
import { SubscriptionTab } from "./components/SubscriptionTab/SubscriptionTab";

export default function SettingsBillingPage() {
  useEffect(function setBillingDocumentTitle() {
    document.title = "Billing – AutoGPT Platform";
  }, []);

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

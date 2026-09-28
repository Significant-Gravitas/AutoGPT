import type { ReactNode } from "react";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Text } from "@/components/atoms/Text/Text";
import type { SubscriptionPlansProps } from "../helpers";
import { BillingToggle } from "./BillingToggle";

type Props = Pick<
  SubscriptionPlansProps,
  "billing" | "onBillingChange" | "goalSurface"
> & {
  // A surface with its own title (the platform paywall modal) passes its
  // heading here, or `null` to keep the heading it already renders outside the
  // organism. Left undefined, the default onboarding heading renders.
  header?: ReactNode;
};

export function PlanHeader({ header, ...billing }: Props) {
  return (
    <header className="mb-5 flex flex-col items-center text-center">
      {header === undefined ? <DefaultHeading /> : header}
      <BillingToggle {...billing} />
    </header>
  );
}

function DefaultHeading() {
  return (
    <>
      <AutoGPTLogo hideText className="relative right-5 mb-2 h-8 w-20" />
      <Text variant="h3" as="h1" className="leading-9">
        Choose the plan that&apos;s right for{" "}
        <span className="bg-gradient-to-r from-purple-500 to-indigo-500 bg-clip-text text-transparent">
          you
        </span>
      </Text>
      <Text variant="body" tone="muted" className="mt-1">
        Upgrade, downgrade, or change plans anytime. All plans include core
        features to get you started.
      </Text>
    </>
  );
}

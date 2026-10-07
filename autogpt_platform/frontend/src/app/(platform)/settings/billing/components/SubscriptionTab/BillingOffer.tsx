"use client";

import { SparklesIcon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { SupportOffer } from "./SupportOffer";
import { useProActivation } from "@/services/pro-activation/useProActivation";
import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";
import { formatTrialPrice } from "@/components/organisms/TrialCard/helpers";
import { formatCents } from "../../helpers";
import {
  ENTERPRISE_CONTACT_URL,
  TEAM_UPGRADE_URL,
  useYourPlanCard,
} from "./YourPlanCard/useYourPlanCard";

interface Props {
  controller?: ReturnType<typeof useYourPlanCard>;
}

export function BillingOffer({ controller }: Props) {
  const { experience, trial, subscription, isError } = useUsageExperience();
  const activation = useProActivation();
  const isTrial = experience.isActiveTrial;
  const paymentFailed =
    experience.tier === "NO_TIER" &&
    !trial?.active &&
    !trial?.converted &&
    ["past_due", "unpaid"].includes(trial?.status ?? "");
  const target = isTrial
    ? trial?.offer?.tier === "PRO"
      ? "Pro"
      : null
    : controller?.plan?.nextTierLabel;
  const canUpgrade = isTrial
    ? target === "Pro"
    : controller?.canUpgrade && !controller.plan?.nextTierIsTeamLink;
  if (activation.isReady && experience.tier === "PRO")
    return (
      <SupportOffer
        title="You’re ready to go."
        description="Pro is active. Pick up where you left off with your conversations and agents."
        label="Back to your work"
        href="/copilot"
        ready
      />
    );
  if (paymentFailed)
    return (
      <SupportOffer
        title="Let’s get you back to work."
        description="Your trial payment needs attention. Review your payment details to continue with your saved work."
        label="Update payment method"
        onAction={controller?.onManage}
        disabled={!controller?.canManagePortal}
      />
    );
  if (isError && experience.tier !== "NO_TIER")
    return (
      <SupportOffer
        title="A little help, if you need it."
        description="We’re having trouble loading your usage. If it keeps happening, our team can help."
        label="Contact support"
        href={ENTERPRISE_CONTACT_URL}
      />
    );
  if (!canUpgrade || !target)
    return (
      <SupportOffer
        title="Room for bigger ambitions."
        description={
          experience.tier === "ENTERPRISE"
            ? "For additional capacity or changes to your plan, your account team is ready to help."
            : "When your work calls for more capacity, talk with us about what you need."
        }
        label={
          experience.tier === "ENTERPRISE"
            ? "Contact account team"
            : "Contact us"
        }
        href={
          experience.tier === "MAX" ? TEAM_UPGRADE_URL : ENTERPRISE_CONTACT_URL
        }
      />
    );
  const price =
    isTrial && trial?.offer
      ? formatTrialPrice(trial.offer)
      : controller?.offerPriceCents !== undefined
        ? `${formatCents(controller.offerPriceCents)} / ${controller.plan?.billingCycle === "yearly" ? "year" : "month"}`
        : null;
  const multipliers = subscription?.tier_multipliers;
  const multiplier =
    multipliers?.MAX && multipliers?.PRO
      ? Math.round((multipliers.MAX / multipliers.PRO) * 10) / 10
      : null;
  const benefits = isTrial
    ? [
        "Fresh usage when Pro activates",
        "Your conversations and agents stay saved",
      ]
    : target === "Max"
      ? [
          multiplier
            ? `${multiplier}× the usage of Pro`
            : "More room for demanding work",
          "More workspace storage and priority support",
        ]
      : [
          "Daily and weekly usage allowances",
          "Your conversations and agents stay saved",
        ];
  return (
    <section
      aria-label={`AutoGPT ${target}`}
      className="flex h-full flex-col rounded-2xl border border-purple-200 bg-gradient-to-br from-purple-50 via-white to-blue-50 p-6"
    >
      <div className="flex items-center gap-2.5">
        <span className="flex size-7 items-center justify-center rounded-lg bg-purple-100 text-purple-700">
          <Icon icon={SparklesIcon} size={17} />
        </span>
        <Text variant="small-medium" className="text-purple-800">
          AutoGPT {target}
        </Text>
      </div>
      <Text
        variant="h4"
        as="h2"
        className="mb-3 mt-4 text-[23px] leading-8 tracking-[-0.04em]"
      >
        {target === "Pro" ? "Keep your momentum." : "Think bigger. Go further."}
      </Text>
      {price && (
        <div className="flex flex-wrap items-baseline gap-1.5">
          <Text
            variant="h2"
            as="span"
            className="font-sans text-[36px] font-semibold leading-10 tracking-[-0.045em]"
          >
            {price.split(" / ")[0]}
          </Text>
          <Text as="span" variant="small" tone="secondary">
            / {price.split(" / ")[1]}
          </Text>
        </div>
      )}
      <ul className="my-5 space-y-2.5">
        {benefits.map((benefit) => (
          <li key={benefit} className="flex items-start gap-2">
            <Icon
              icon={Tick02Icon}
              size={15}
              className="mt-0.5 shrink-0 text-purple-700"
            />
            <Text variant="small" className="text-zinc-700">
              {benefit}
            </Text>
          </li>
        ))}
      </ul>
      <Button
        size="large"
        className="mt-auto w-full"
        onClick={
          isTrial
            ? () => activation.start("/settings/billing")
            : controller?.onUpgrade
        }
        disabled={isTrial ? activation.isBusy : controller?.isUpdatingTier}
      >
        Upgrade to {target}
      </Button>
      <Text
        variant="small"
        tone="secondary"
        className="mt-2.5 text-center text-xs leading-5"
      >
        {isTrial
          ? "Usage refreshes after successful payment."
          : target === "Max"
            ? "Your current usage carries over."
            : "Review your options before paying."}
      </Text>
    </section>
  );
}

"use client";

import { BillingPlanOffer } from "./BillingPlanOffer";
import { SupportOffer } from "./SupportOffer";
import { useProActivation } from "@/services/pro-activation/useProActivation";
import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";
import {
  formatOfferPrice,
  formatTrialStartDescription,
} from "@/services/usageExperience/helpers";
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
        eyebrow="AutoGPT Pro"
        title="Pro is active"
        description="Pro is active. Pick up where you left off with your conversations and agents."
        label="Back to your work"
        href="/copilot"
        ready
      />
    );
  if (paymentFailed)
    return (
      <SupportOffer
        eyebrow="Payment"
        footer="Your conversations and agents stay saved."
        title="Your payment didn’t go through"
        description="Your trial payment needs attention. Review your payment details to continue with your saved work."
        label="Update payment method"
        onAction={controller?.onManage}
        disabled={!controller?.canManagePortal}
      />
    );
  if (isError && experience.tier !== "NO_TIER")
    return (
      <SupportOffer
        eyebrow="Usage"
        title="We couldn’t load your usage"
        description="We’re having trouble loading your usage. If it keeps happening, our team can help."
        label="Contact support"
        href={ENTERPRISE_CONTACT_URL}
      />
    );
  if (!canUpgrade || !target)
    return (
      <SupportOffer
        eyebrow="Your plan"
        title="Need more capacity?"
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
      ? `${formatOfferPrice(trial.offer.unit_amount, trial.offer.currency)} / ${trial.offer.billing_cycle === "yearly" ? "year" : "month"}`
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
        "A full daily and weekly allowance once payment goes through",
        "Your chats, agents and files stay where they are",
      ]
    : target === "Max"
      ? [
          multiplier
            ? `${multiplier}× the usage of Pro`
            : "More room for demanding work",
          "More workspace storage and priority support",
        ]
      : [
          "Experts for content, sales and research, with Otto coordinating",
          "Agents that keep working in the background on a schedule",
        ];
  const title = isTrial
    ? "Start Pro now"
    : `${experience.tier === "NO_TIER" ? "Start" : "Upgrade to"} ${target}`;
  return (
    <BillingPlanOffer
      target={target}
      title={title}
      description={
        isTrial && experience.freshProUsage
          ? formatTrialStartDescription(experience.trialEndsAt)
          : undefined
      }
      price={price}
      benefits={benefits}
      actionLabel={
        isTrial && price ? `${title} · ${price.split(" / ")[0]}` : title
      }
      onAction={
        isTrial
          ? () => activation.start("/settings/billing")
          : controller?.onUpgrade
      }
      disabled={isTrial ? activation.isBusy : controller?.isUpdatingTier}
      caption={
        isTrial
          ? "Starting now ends your trial early. You’ll review the charge first."
          : target === "Max"
            ? "What you’ve used today counts toward Max’s higher limit."
            : "Review your options before paying."
      }
    />
  );
}

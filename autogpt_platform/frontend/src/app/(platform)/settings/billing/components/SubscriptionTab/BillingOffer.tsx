"use client";

import { BillingPlanOffer } from "./BillingPlanOffer";
import { SupportOffer } from "./SupportOffer";
import { useProActivation } from "@/services/pro-activation/useProActivation";
import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";
import {
  formatOfferPrice,
  formatTrialStartDescription,
  formatTrialStartTitle,
  getFeaturedUsagePercent,
  getUsageStatus,
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
  const { experience, trial, subscription, usage } = useUsageExperience();
  const activation = useProActivation();
  const isTrial = experience.isActiveTrial;
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
        description="Pro is active. Pick up where you left off with your conversations and workflows."
        label="Back to your work"
        href="/copilot"
        ready
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
        "Your chats, workflows and files stay where they are",
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
          "Workflows that keep working in the background on a schedule",
        ];
  const quiet =
    isTrial &&
    getUsageStatus(getFeaturedUsagePercent(experience, usage)) === "plenty";
  const title = quiet
    ? formatTrialStartTitle(experience.trialEndsAt)
    : isTrial
      ? "Start Pro now"
      : `${experience.tier === "NO_TIER" ? "Start" : "Upgrade to"} ${target}`;
  return (
    <BillingPlanOffer
      target={target}
      title={title}
      quiet={quiet}
      description={
        isTrial && experience.freshProUsage
          ? quiet
            ? "Your trial keeps going until then. You can also start today."
            : formatTrialStartDescription(experience.trialEndsAt)
          : undefined
      }
      price={price}
      benefits={benefits}
      actionLabel={
        quiet
          ? "Start Pro now"
          : isTrial && price
            ? `${title} · ${price.split(" / ")[0]}`
            : title
      }
      onAction={
        isTrial
          ? () => activation.start("/settings/billing")
          : controller?.onUpgrade
      }
      disabled={
        isTrial
          ? activation.isBusy
          : controller?.isUpdatingTier || controller?.isRedirecting
      }
      caption={
        isTrial
          ? "Starting now ends your trial early. You’ll review the charge first."
          : target === "Max"
            ? "What you’ve already used counts toward Max’s higher limit."
            : "Review your options before paying."
      }
    />
  );
}

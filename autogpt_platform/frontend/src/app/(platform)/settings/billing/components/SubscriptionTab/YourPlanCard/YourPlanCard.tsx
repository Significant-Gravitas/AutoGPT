"use client";

import { motion, useReducedMotion } from "framer-motion";
import { ArrowSquareOutIcon } from "@phosphor-icons/react";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";

import {
  formatCents,
  formatShortDate,
  getSectionMotionProps,
} from "../../../helpers";
import { useYourPlanCard } from "./useYourPlanCard";

const PRICING_PAGE_URL = "https://agpt.co/pricing";

interface Props {
  index?: number;
}

export function YourPlanCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const {
    plan,
    isLoading,
    isUpdatingTier,
    canManagePortal,
    canUpgrade,
    canDowngrade,
    canResume,
    onUpgrade,
    onDowngrade,
    onResume,
    onManage,
  } = useYourPlanCard();

  const sectionMotion = getSectionMotionProps(index, Boolean(reduceMotion));

  if (isLoading || !plan) {
    return (
      <motion.div {...sectionMotion}>
        <Skeleton className="h-[100px] rounded-[18px]" />
      </motion.div>
    );
  }

  return (
    <motion.section {...sectionMotion} className="flex w-full flex-col gap-2">
      <div className="flex items-center gap-2 px-4">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Your plan
        </Text>
        <a
          href={PRICING_PAGE_URL}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Compare plans on the AutoGPT pricing page"
          className="inline-flex items-center gap-1 text-zinc-500 hover:text-violet-700"
        >
          <Text variant="small" as="span">
            Compare plans
          </Text>
          <ArrowSquareOutIcon size={14} aria-hidden="true" />
        </a>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-4 rounded-[18px] border border-zinc-200 bg-white p-5 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex items-center gap-2">
            <Text variant="large-medium" as="span" className="text-textBlack">
              {plan.label}
            </Text>
            <Badge
              variant="success"
              size="small"
              className={
                plan.isPendingCancel || plan.isPendingDowngrade
                  ? "bg-amber-100 text-amber-800"
                  : "bg-violet-100 text-violet-800"
              }
            >
              {!plan.isPaidPlan
                ? "Inactive"
                : plan.isPendingCancel
                  ? "Cancellation scheduled"
                  : plan.isPendingDowngrade
                    ? "Downgrade scheduled"
                    : "Active"}
            </Badge>
          </div>
          {plan.isPaidPlan ? (
            <Text variant="body" as="span" className="text-zinc-700">
              {formatCents(plan.monthlyCostCents)} / month
              {plan.isPendingCancel && plan.pendingEffectiveAt
                ? ` · Ends on ${formatShortDate(plan.pendingEffectiveAt)}`
                : plan.isPendingDowngrade && plan.pendingEffectiveAt
                  ? ` · Switches to ${plan.pendingTierLabel} on ${formatShortDate(plan.pendingEffectiveAt)}`
                  : plan.currentPeriodEnd
                    ? ` · Renews on ${formatShortDate(plan.currentPeriodEnd * 1000)}`
                    : null}
            </Text>
          ) : (
            <Text variant="body" as="span" className="text-zinc-700">
              Pick a plan to continue using AutoGPT.
            </Text>
          )}
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          {canResume ? (
            <Button
              variant="primary"
              size="small"
              onClick={onResume}
              disabled={isUpdatingTier}
              loading={isUpdatingTier}
            >
              {plan.isPendingCancel
                ? "Resume subscription"
                : "Cancel downgrade"}
            </Button>
          ) : null}
          {canDowngrade && plan.previousTierLabel ? (
            <Button
              variant="outline"
              size="small"
              onClick={onDowngrade}
              disabled={isUpdatingTier}
              loading={isUpdatingTier}
            >
              Downgrade to {plan.previousTierLabel}
            </Button>
          ) : null}
          {plan.isPaidPlan ? (
            <Button
              variant="secondary"
              size="small"
              onClick={onManage}
              disabled={!canManagePortal}
            >
              Manage subscription
            </Button>
          ) : null}
          {canUpgrade && plan.nextTierLabel ? (
            <Button
              variant="primary"
              size="small"
              onClick={onUpgrade}
              disabled={isUpdatingTier}
              loading={isUpdatingTier && !plan.nextTierIsTeamLink}
              rightIcon={
                plan.nextTierIsTeamLink ? (
                  <ArrowSquareOutIcon size={14} aria-hidden="true" />
                ) : undefined
              }
            >
              {!plan.isPaidPlan
                ? `Get ${plan.nextTierLabel}`
                : plan.nextTierIsTeamLink
                  ? `Talk to sales — ${plan.nextTierLabel}`
                  : `Upgrade to ${plan.nextTierLabel}`}
            </Button>
          ) : null}
        </div>
      </div>
    </motion.section>
  );
}

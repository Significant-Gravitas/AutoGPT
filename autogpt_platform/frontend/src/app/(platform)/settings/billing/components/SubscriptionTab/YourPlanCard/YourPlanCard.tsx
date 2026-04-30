"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";

import { EASE_OUT, formatCents, formatShortDate } from "../../../helpers";
import { useYourPlanCard } from "./useYourPlanCard";

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
    onUpgrade,
    onCancel,
    onManage,
  } = useYourPlanCard();

  if (isLoading || !plan) {
    return <Skeleton className="h-[100px] rounded-[18px]" />;
  }

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : { duration: 0.32, ease: EASE_OUT, delay: 0.04 + index * 0.05 }
      }
      className="flex w-full flex-col gap-2"
    >
      <div className="px-4">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Your plan
        </Text>
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
              className="bg-violet-100 text-violet-800"
            >
              {plan.isPaidPlan ? "Active" : "No active subscription"}
            </Badge>
          </div>
          {plan.isPaidPlan ? (
            <Text variant="body" as="span" className="text-zinc-700">
              {formatCents(plan.monthlyCostCents)} / month
              {plan.currentPeriodEnd
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
          {plan.isPaidPlan ? (
            <Button
              variant="ghost"
              size="small"
              onClick={onCancel}
              disabled={isUpdatingTier || !canManagePortal}
            >
              Cancel plan
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
          {canUpgrade ? (
            <Button
              variant="primary"
              size="small"
              onClick={onUpgrade}
              disabled={isUpdatingTier}
            >
              {plan.isPaidPlan ? "Upgrade plan" : "Choose a plan"}
            </Button>
          ) : null}
        </div>
      </div>
    </motion.section>
  );
}

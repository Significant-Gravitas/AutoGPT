"use client";

import { useState } from "react";

import { useUsageActions } from "@/services/usageExperience/useUsageActions";

import type { TurnPause } from "../../../../turnPause";
import { useProviderLimitDialog } from "../../../ProviderLimitDialog/useProviderLimitDialog";
import {
  hasReset,
  isBudgetPause,
  neverResets,
  pauseBody,
  pauseTitle,
} from "../../helpers";
import {
  canContinueTask,
  pauseAfterRefusal,
  PLATFORM_BUDGET,
  primaryAction,
} from "./helpers";
import { useKeepGoing } from "./useKeepGoing";
import { useNow } from "./useNow";
import { useResumeAtReset } from "./useResumeAtReset";

interface Args {
  pause: TurnPause;
  sessionId: string;
  assistantName: string;
  onFollow: () => void;
}

export function useLivePausedCard({
  pause,
  sessionId,
  assistantName,
  onFollow,
}: Args) {
  const usage = useUsageActions();
  const { experience } = usage;
  // Set when "Keep going" was refused because the account ran out meanwhile.
  const [refusal, setRefusal] = useState<{ resetsAt: string | null }>();
  const [isBudgetBack, setIsBudgetBack] = useState(false);
  const shown = refusal
    ? pauseAfterRefusal(
        pause,
        experience.window,
        refusal.resetsAt ?? isoOrNull(experience.resetsAt),
      )
    : pause;
  const isBudget = isBudgetPause(shown);
  const waitsOnReset = isBudget && !neverResets(shown);
  // Once the budget is back the card's text stops changing.
  const now = useNow(
    waitsOnReset && !isBudgetBack && shown.resets_at
      ? Date.parse(shown.resets_at)
      : null,
  );
  const budgetBack = isBudget && (isBudgetBack || hasReset(shown, now));
  const keep = useKeepGoing({
    sessionId,
    onFollow,
    onOutOfBudget: (resetsAt) => {
      setRefusal({ resetsAt });
      setIsBudgetBack(false);
    },
  });
  const resume = useResumeAtReset({
    sessionId,
    enabled: waitsOnReset,
    onBudgetBack: () => setIsBudgetBack(true),
  });
  // Switching the chat's connection, then picking the task up on it.
  const chatGpt = useProviderLimitDialog({
    failure: isBudget ? PLATFORM_BUDGET : null,
    sessionId,
    onDismiss: () => void keep.keepGoing(),
    eligible: canContinueTask,
  });

  return {
    shown,
    now,
    title: pauseTitle(shown),
    body: pauseBody({ pause: shown, name: assistantName, budgetBack, now }),
    primary: primaryAction({ pause: shown, budgetBack, resume }),
    canStop: shown.reason === "task" || shown.reason === "steps",
    canUpgrade:
      (isBudget || shown.reason === "trial") &&
      usage.isBillingEnabled &&
      experience.targetTier !== null,
    chatGptName: isBudget ? (chatGpt.alternative?.display_name ?? null) : null,
    keep,
    resume,
    chatGpt,
    upgrade: usage.upgrade,
    isUpgradeDisabled: usage.offer.disabled,
  };
}

function isoOrNull(value: string | Date | null | undefined): string | null {
  const date = value ? new Date(value) : null;
  return date && Number.isFinite(date.getTime()) ? date.toISOString() : null;
}

import type { SubscriptionStatusResponseTier } from "@/app/api/__generated__/models/subscriptionStatusResponseTier";
import { formatShortDate } from "@/app/(platform)/settings/billing/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { PLAN_LABEL } from "@/services/billing/subscription-plan";

export type ConsequenceKey = "data" | "apps" | "plan" | "listings";

export interface Consequence {
  key: ConsequenceKey;
  text: string;
}

interface ConsequencesArgs {
  appNames: string[];
  planLabel: string | null;
  publishedWorkflows: boolean;
  publishedSkills: boolean;
}

interface EmailMatchArgs {
  typed: string;
  email: string;
}

const DATA_LINE =
  "Your chats, Experts, workflows, files and memory are erased after 14 days. Your share links stop working and your Marketplace reviews are removed now.";
const TRIAL_LABEL = "trial";
const TRIAL_LINE = "Your trial ends now.";
const FALLBACK_ERROR = "We couldn't delete your account. Try again.";

export function getConsequences({
  appNames,
  planLabel,
  publishedWorkflows,
  publishedSkills,
}: ConsequencesArgs): Consequence[] {
  const lines: Consequence[] = [{ key: "data", text: DATA_LINE }];
  if (appNames.length > 0) {
    lines.push({
      key: "apps",
      text: `Connected apps like ${formatAppsLike(appNames)} are disconnected, and our access to them is revoked.`,
    });
  }
  if (planLabel) {
    lines.push({
      key: "plan",
      text:
        planLabel === TRIAL_LABEL
          ? TRIAL_LINE
          : `Your ${planLabel} plan ends now. The rest of this billing period isn't refunded.`,
    });
  }
  const listings = getListingsLine({ publishedWorkflows, publishedSkills });
  if (listings) lines.push({ key: "listings", text: listings });
  return lines;
}

function getListingsLine({
  publishedWorkflows,
  publishedSkills,
}: Pick<ConsequencesArgs, "publishedWorkflows" | "publishedSkills">) {
  if (publishedWorkflows && publishedSkills)
    return "Your published workflows and skills are removed from the Marketplace.";
  if (publishedWorkflows)
    return "Your published workflows are removed from the Marketplace.";
  if (publishedSkills)
    return "Your published skills are removed from the Marketplace.";
  return null;
}

export function formatAppsLike(names: string[]): string {
  if (names.length > 2) return `${names[0]}, ${names[1]} and others`;
  return names.join(" and ");
}

export function getPlanLabel(
  tier: SubscriptionStatusResponseTier | null,
): string | null {
  if (!tier || tier === "NO_TIER") return null;
  if (tier === "TRIAL") return TRIAL_LABEL;
  return PLAN_LABEL[tier] ?? null;
}

export function emailMatches({ typed, email }: EmailMatchArgs): boolean {
  const expected = normalizeEmail(email);
  return expected !== "" && normalizeEmail(typed) === expected;
}

export function getClosedToastTitle(
  eraseAfter: Date | string | null | undefined,
): string {
  return eraseAfter
    ? `Your account is closed. Sign in before ${formatShortDate(eraseAfter)} if you change your mind.`
    : "Your account is closed. Sign in within 14 days if you change your mind.";
}

export function getDeletionErrorMessage(error: unknown): string {
  return error instanceof ApiError && error.message
    ? error.message
    : FALLBACK_ERROR;
}

function normalizeEmail(value: string): string {
  return value.trim().toLowerCase();
}

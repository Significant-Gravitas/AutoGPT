import type { SubscriptionStatusResponseTier } from "@/app/api/__generated__/models/subscriptionStatusResponseTier";
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
  hasListings: boolean;
}

interface EmailMatchArgs {
  typed: string;
  email: string;
}

const DATA_LINE =
  "Your chats, Experts, workflows, files and memory are deleted.";
const LISTINGS_LINE =
  "Your published workflows are removed from the Marketplace.";
const FALLBACK_ERROR = "We couldn't delete your account. Try again.";

export function getConsequences({
  appNames,
  planLabel,
  hasListings,
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
      text: `Your ${planLabel} plan is canceled and won't renew.`,
    });
  }
  if (hasListings) lines.push({ key: "listings", text: LISTINGS_LINE });
  return lines;
}

export function formatAppsLike(names: string[]): string {
  if (names.length > 2) return `${names[0]}, ${names[1]} and others`;
  return names.join(" and ");
}

export function getPlanLabel(
  tier: SubscriptionStatusResponseTier | null,
): string | null {
  if (!tier || tier === "NO_TIER") return null;
  if (tier === "TRIAL") return "trial";
  return PLAN_LABEL[tier] ?? null;
}

export function emailMatches({ typed, email }: EmailMatchArgs): boolean {
  const expected = normalizeEmail(email);
  return expected !== "" && normalizeEmail(typed) === expected;
}

export function getDeletionErrorMessage(error: unknown): string {
  return error instanceof ApiError && error.message
    ? error.message
    : FALLBACK_ERROR;
}

function normalizeEmail(value: string): string {
  return value.trim().toLowerCase();
}

import type { SubscriptionStatusResponseTier } from "@/app/api/__generated__/models/subscriptionStatusResponseTier";

export type DeletionSource = "settings" | "paywall";

export type ExitReason =
  | "privacy"
  | "cost"
  | "not_using"
  | "missing_capability"
  | "limits"
  | "other";

export type PlanState = "max" | "pro" | "trial" | "none" | "managed";

export type PanelActionKind =
  | "manage_apps"
  | "disconnect_all"
  | "billing"
  | "keep_account";

export interface PanelAction {
  kind: PanelActionKind;
  label: string;
}

export interface ReasonPanel {
  copy: string;
  action: PanelAction | null;
}

export interface PendingChange {
  kind: "cancel" | "downgrade";
  effectiveAt: string | number | Date;
}

export type AppsState =
  | { status: "ready"; names: string[] }
  | { status: "unavailable" };

export type DisconnectState =
  | { status: "idle" }
  | { status: "done"; names: string[] }
  | { status: "failed"; names: string[] };

interface PlanStateArgs {
  tier: SubscriptionStatusResponseTier | null | undefined;
  isAdminManaged: boolean;
}

const PLAN_STATE_BY_TIER: Record<SubscriptionStatusResponseTier, PlanState> = {
  MAX: "max",
  PRO: "pro",
  TRIAL: "trial",
  NO_TIER: "none",
  BASIC: "managed",
  BUSINESS: "managed",
  ENTERPRISE: "managed",
};

export function getPlanState({
  tier,
  isAdminManaged,
}: PlanStateArgs): PlanState | null {
  if (tier === undefined || tier === null) return null;
  if (isAdminManaged) return "managed";
  return PLAN_STATE_BY_TIER[tier] ?? null;
}

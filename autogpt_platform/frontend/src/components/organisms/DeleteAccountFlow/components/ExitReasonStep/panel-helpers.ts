import {
  formatCents,
  formatShortDate,
} from "@/app/(platform)/settings/billing/helpers";
import type {
  AppsState,
  DeletionSource,
  DisconnectState,
  ExitReason,
  PanelAction,
  PanelActionKind,
  PendingChange,
  PlanState,
  ReasonPanel,
} from "../../helpers";

interface ReasonPanelArgs {
  reason: ExitReason | null;
  planState: PlanState | null;
  pending: PendingChange | null;
  apps: AppsState;
  proMonthlyCents: number | null;
  trialEndsAt: PendingChange["effectiveAt"] | null;
  origin: DeletionSource;
  disconnect: DisconnectState;
}

const ACTIONS: Record<PanelActionKind, PanelAction> = {
  manage_apps: { kind: "manage_apps", label: "Manage connected apps" },
  disconnect_all: { kind: "disconnect_all", label: "Disconnect all apps" },
  billing: { kind: "billing", label: "Go to Billing" },
  keep_account: { kind: "keep_account", label: "Keep my account" },
};

const NO_APPS_COPY =
  "You haven't connected any apps, so AutoGPT can't reach your email, calendar or files.";
const PRO_CANCEL_COPY =
  "You can cancel your plan and keep your account. You won't be charged again, and your Experts and chats stay here if you come back.";
const TRIAL_CANCEL_COPY =
  "You can cancel your trial instead. You won't be charged.";
const NO_PLAN_COPY =
  "Your account costs nothing without a plan. Your Experts and chats stay here if you come back.";
const MAX_DOWNGRADE_COPY =
  "A downgrade keeps your Experts, chats and schedules and starts when your current period ends.";
const LIMITS_COPY = "Max gives you 8.5x the usage of Pro.";

const NAMED_APPS = 2;

export function getReasonPanel(args: ReasonPanelArgs): ReasonPanel | null {
  switch (args.reason) {
    case "privacy":
      return getPrivacyPanel(args);
    case "cost":
      return getCostPanel(args);
    case "not_using":
      return getNotUsingPanel(args);
    case "limits":
      return getLimitsPanel(args);
    case "missing_capability":
    case "other":
    case null:
      return null;
  }
}

export function formatAppNames(names: string[]): string {
  if (names.length <= NAMED_APPS) return names.join(" and ");
  const more = names.length - NAMED_APPS;
  return `${names[0]}, ${names[1]} and ${more} more ${more === 1 ? "app" : "apps"}`;
}

function getPrivacyPanel({
  apps,
  origin,
  disconnect,
}: ReasonPanelArgs): ReasonPanel | null {
  if (disconnect.status !== "idle" && disconnect.names.length > 0) {
    return getDisconnectResultPanel(disconnect);
  }
  if (apps.status === "unavailable") return null;
  if (apps.names.length === 0) return { copy: NO_APPS_COPY, action: null };
  return {
    copy: `You can disconnect ${formatAppNames(apps.names)} without deleting your account. Our access ends as soon as you do.`,
    action: origin === "paywall" ? ACTIONS.disconnect_all : ACTIONS.manage_apps,
  };
}

function getDisconnectResultPanel(
  disconnect: Exclude<DisconnectState, { status: "idle" }>,
): ReasonPanel {
  const appNames = formatAppNames(disconnect.names);
  if (disconnect.status === "done") {
    const verb = disconnect.names.length === 1 ? "is" : "are";
    return { copy: `${appNames} ${verb} disconnected.`, action: null };
  }
  return {
    copy: `Couldn't disconnect ${appNames}. Try again.`,
    action: ACTIONS.disconnect_all,
  };
}

function getCostPanel(args: ReasonPanelArgs): ReasonPanel | null {
  const { planState, pending } = args;
  switch (planState) {
    case "max":
      return getMaxCostPanel(args);
    case "pro":
      return pending?.kind === "cancel"
        ? getPlanEndsPanel(pending)
        : { copy: PRO_CANCEL_COPY, action: ACTIONS.billing };
    case "trial":
      return pending?.kind === "cancel"
        ? getTrialEndsPanel(pending)
        : getTrialCancelPanel(args);
    case "none":
      return { copy: NO_PLAN_COPY, action: ACTIONS.keep_account };
    case "managed":
    case null:
      return null;
  }
}

function getNotUsingPanel(args: ReasonPanelArgs): ReasonPanel | null {
  if (args.planState === "max" && args.pending?.kind !== "cancel") {
    return { copy: PRO_CANCEL_COPY, action: ACTIONS.billing };
  }
  return getCostPanel(args);
}

function getLimitsPanel({
  planState,
  pending,
}: ReasonPanelArgs): ReasonPanel | null {
  if (planState !== "pro" || pending?.kind === "cancel") return null;
  return { copy: LIMITS_COPY, action: ACTIONS.billing };
}

function getMaxCostPanel({
  pending,
  proMonthlyCents,
}: ReasonPanelArgs): ReasonPanel {
  if (pending?.kind === "cancel") return getPlanEndsPanel(pending);
  if (pending?.kind === "downgrade") {
    return {
      copy: `Your switch to Pro is already scheduled for ${formatShortDate(pending.effectiveAt)}.`,
      action: null,
    };
  }
  if (proMonthlyCents === null || proMonthlyCents <= 0) {
    return { copy: MAX_DOWNGRADE_COPY, action: ACTIONS.billing };
  }
  return {
    copy: `Pro is ${formatCents(proMonthlyCents)} / month. ${MAX_DOWNGRADE_COPY}`,
    action: ACTIONS.billing,
  };
}

function getPlanEndsPanel(pending: PendingChange): ReasonPanel {
  return {
    copy: `Your plan already ends on ${formatShortDate(pending.effectiveAt)}, and you won't be charged again. Your account stays after that.`,
    action: ACTIONS.keep_account,
  };
}

function getTrialCancelPanel({ trialEndsAt }: ReasonPanelArgs): ReasonPanel {
  return {
    copy: trialEndsAt
      ? `You can cancel your trial instead. You won't be charged and you keep access until ${formatShortDate(trialEndsAt)}.`
      : TRIAL_CANCEL_COPY,
    action: ACTIONS.billing,
  };
}

function getTrialEndsPanel(pending: PendingChange): ReasonPanel {
  return {
    copy: `Your trial already ends on ${formatShortDate(pending.effectiveAt)} and you won't be charged.`,
    action: ACTIONS.keep_account,
  };
}

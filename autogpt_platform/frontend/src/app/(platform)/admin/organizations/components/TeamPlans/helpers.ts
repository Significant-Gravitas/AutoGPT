import { getGetAdminListAnAccountSTeamOrganizationsQueryKey } from "@/app/api/__generated__/endpoints/admin/admin";
import type { OrgPlanSummary } from "@/app/api/__generated__/models/orgPlanSummary";
import type { OrgPlanView } from "@/app/api/__generated__/models/orgPlanView";
import type { QueryClient } from "@tanstack/react-query";
import { pluralize } from "../../../users/helpers";

// Plan dates are read and written as UTC days, so "31 Jan 2027" means the
// same day to every admin whatever their time zone.
const PLAN_DATE_FORMAT = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});
const DOLLAR_FORMAT = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});
const STATUS_LABELS: Record<string, string> = {
  active: "Active",
  canceled: "Canceled",
};

// A plan counts until ends_at; its last day is the UTC day just before.
function formatLastDay(endsAt: Date | string) {
  return PLAN_DATE_FORMAT.format(new Date(new Date(endsAt).getTime() - 1));
}

export function formatCredits(credits: number) {
  return pluralize(credits, "credit");
}

// 100 credits are $1.
export function formatBalance(credits: number) {
  return `${formatCredits(credits)} (${DOLLAR_FORMAT.format(credits / 100)})`;
}

export function getStatusLabel(status: string) {
  return STATUS_LABELS[status] ?? status;
}

// The server's in_effect decides; the clock only picks "Runs through" or
// "Ended".
function hasEnded(plan: OrgPlanView, now = new Date()) {
  return (
    !plan.in_effect &&
    plan.ends_at !== null &&
    new Date(plan.ends_at).getTime() <= now.getTime()
  );
}

export function describeEnd(plan: OrgPlanView, now = new Date()) {
  if (plan.ends_at === null) return "No end date";
  const lastDay = formatLastDay(plan.ends_at);
  return hasEnded(plan, now) ? `Ended ${lastDay}` : `Runs through ${lastDay}`;
}

// Without a plan in effect, members get the paywall in the organization and
// the owner holds its only seat (backend/api/features/orgs/seats.py).
export function describeNotInEffect(
  plan: OrgPlanView | null,
  now = new Date(),
) {
  const reason = getNotInEffectReason(plan, now);
  if (reason === null) return null;
  return `${reason}, so members see the paywall in this organization and nobody can be invited.`;
}

function getNotInEffectReason(plan: OrgPlanView | null, now: Date) {
  if (plan === null) return "There's no plan";
  if (plan.in_effect) return null;
  if (plan.status === "canceled") return "The plan is canceled";
  if (plan.status !== "active") {
    return `The plan's status is ${getStatusLabel(plan.status)}`;
  }
  if (plan.ends_at !== null && hasEnded(plan, now)) {
    return `The plan ended on ${formatLastDay(plan.ends_at)}`;
  }
  return "The plan's tier isn't a paid tier";
}

export function describeSeatsInUse(org: OrgPlanSummary) {
  const people = [pluralize(org.members, "member")];
  if (org.pending_invitations > 0) {
    people.push(pluralize(org.pending_invitations, "pending invitation"));
  }
  const inUse = org.seats_in_use.toLocaleString("en-GB");
  const count = org.plan
    ? `${inUse} of ${org.plan.seats.toLocaleString("en-GB")}`
    : inUse;
  return `${count} in use (${people.join(", ")})`;
}

export function isOverSeats(org: OrgPlanSummary) {
  return org.plan !== null && org.seats_in_use > org.plan.seats;
}

export function getErrorStatus(error: unknown) {
  return typeof error === "object" && error !== null && "status" in error
    ? (error as { status: unknown }).status
    : undefined;
}

// A 4xx answers the same on every try, so only other failures are retried.
export function shouldRetryLookup(failureCount: number, error: unknown) {
  const status = getErrorStatus(error);
  const isClientError =
    typeof status === "number" && status >= 400 && status < 500;
  return !isClientError && failureCount < 3;
}

export function refreshTeamOrgs(queryClient: QueryClient) {
  void queryClient.invalidateQueries({
    queryKey: getGetAdminListAnAccountSTeamOrganizationsQueryKey(),
  });
}

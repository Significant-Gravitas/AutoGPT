import type { OrgPlanInput } from "@/app/api/__generated__/models/orgPlanInput";
import { OrgPlanInputStatus } from "@/app/api/__generated__/models/orgPlanInputStatus";
import type { OrgPlanSummary } from "@/app/api/__generated__/models/orgPlanSummary";
import { SubscriptionTier } from "@/app/api/__generated__/models/subscriptionTier";
import { getPlanName, pluralize } from "../../../../../users/helpers";

export const MAX_SEATS = 100_000;
export const NOTE_MAX_LENGTH = 500;

// The tiers a team plan can give; NO_TIER and TRIAL are personal states only.
const PLAN_TIERS: SubscriptionTier[] = [
  SubscriptionTier.BASIC,
  SubscriptionTier.PRO,
  SubscriptionTier.MAX,
  SubscriptionTier.BUSINESS,
  SubscriptionTier.ENTERPRISE,
];

export const TIER_OPTIONS = PLAN_TIERS.map((tier) => ({
  value: tier,
  label: getPlanName(tier),
}));

export const STATUS_OPTIONS = [
  { value: OrgPlanInputStatus.active, label: "Active" },
  { value: OrgPlanInputStatus.canceled, label: "Canceled" },
];

export interface PlanForm {
  tier: SubscriptionTier | "";
  seats: string;
  // "YYYY-MM-DD" from the date field, or "" for no end date.
  endDate: string;
  status: OrgPlanInputStatus;
  note: string;
}

export function getPlanForm(org: OrgPlanSummary): PlanForm {
  const { plan } = org;
  return {
    tier: plan ? toTier(plan.tier) : "",
    seats: String(plan?.seats ?? Math.max(1, org.seats_in_use)),
    endDate: plan?.ends_at ? toDateFieldValue(plan.ends_at) : "",
    status: toStatus(plan?.status ?? OrgPlanInputStatus.active),
    note: plan?.note ?? "",
  };
}

export function toTier(value: string) {
  return PLAN_TIERS.find((tier) => tier === value) ?? "";
}

export function toStatus(value: string) {
  return value === OrgPlanInputStatus.canceled
    ? OrgPlanInputStatus.canceled
    : OrgPlanInputStatus.active;
}

function parseSeats(value: string) {
  if (!/^\d+$/.test(value.trim())) return null;
  const seats = Number(value);
  return seats >= 1 && seats <= MAX_SEATS ? seats : null;
}

// The plan stops at 00:00 UTC on the chosen day.
function parseEndDate(value: string) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const date = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function getSeatsError(form: PlanForm) {
  if (!form.seats.trim() || parseSeats(form.seats) !== null) return undefined;
  return `Enter a whole number from 1 to ${MAX_SEATS.toLocaleString("en-GB")}.`;
}

export function getEndDateError(form: PlanForm) {
  if (!form.endDate || parseEndDate(form.endDate) !== null) return undefined;
  return "Enter a date, or leave it empty.";
}

export function toPlanInput(form: PlanForm): OrgPlanInput | null {
  const seats = parseSeats(form.seats);
  const endsAt = form.endDate ? parseEndDate(form.endDate) : null;
  if (!form.tier || seats === null) return null;
  if (form.endDate && endsAt === null) return null;
  return {
    tier: form.tier,
    seats,
    ends_at: endsAt,
    status: form.status,
    note: form.note.trim() || null,
  };
}

// Saving fewer seats than are taken is allowed, but the organization then
// can't invite anyone, and pending invitations can't be accepted.
export function getSeatsWarning(form: PlanForm, org: OrgPlanSummary) {
  const seats = parseSeats(form.seats);
  if (seats === null || seats >= org.seats_in_use) return null;
  const people = [pluralize(org.members, "member")];
  if (org.pending_invitations > 0) {
    people.push(pluralize(org.pending_invitations, "pending invitation"));
  }
  return (
    `${org.name} already uses ${pluralize(org.seats_in_use, "seat")}: ` +
    `${people.join(" and ")}. Nobody is removed, but with ` +
    `${pluralize(seats, "seat")}, new invitations and acceptances are ` +
    `refused until fewer seats are in use.`
  );
}

function toDateFieldValue(value: Date | string) {
  return new Date(value).toISOString().slice(0, 10);
}

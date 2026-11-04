import type { AddOrgCreditsRequest } from "@/app/api/__generated__/models/addOrgCreditsRequest";
import type { OrgCreditGrantResult } from "@/app/api/__generated__/models/orgCreditGrantResult";
import { formatBalance } from "../../helpers";

const MAX_CREDITS = 100_000_000;
export const REFERENCE_MAX_LENGTH = 200;

function parseCredits(value: string) {
  if (!/^\d+$/.test(value.trim())) return null;
  const credits = Number(value);
  return credits >= 1 && credits <= MAX_CREDITS ? credits : null;
}

export function getAmountError(amount: string) {
  if (!amount.trim() || parseCredits(amount) !== null) return undefined;
  return `Enter a whole number from 1 to ${MAX_CREDITS.toLocaleString("en-GB")}.`;
}

export function toCreditsInput(
  amount: string,
  reference: string,
): AddOrgCreditsRequest | null {
  const credits = parseCredits(amount);
  const trimmed = reference.trim();
  if (credits === null || !trimmed) return null;
  return { amount: credits, reference: trimmed };
}

export function describeRepeat(grant: OrgCreditGrantResult) {
  return (
    `Credits for ${grant.reference} were already added, so nothing ` +
    `changed. The balance is ${formatBalance(grant.balance)}.`
  );
}

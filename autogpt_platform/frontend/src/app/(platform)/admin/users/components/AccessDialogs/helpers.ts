import { pluralize } from "../../helpers";
import type { BanAccountResult } from "../../useBanAccounts";
import type { ChecklistAccount } from "./AccountChecklist";

export const BAN_SENTENCE =
  "Their sessions are signed out. Tokens they hold or get in the next 5 minutes work for up to an hour, and their API keys, schedules and triggers keep running.";

interface TargetArgs {
  accounts: ChecklistAccount[];
  ticked: Set<string>;
  results: Map<string, BanAccountResult>;
}

// After a partial failure only the failed accounts are retried, and only those
// the admin still has ticked.
export function getBanTargets({ accounts, ticked, results }: TargetArgs) {
  const failed = [...results.values()]
    .filter((result) => !result.ok)
    .map((result) => result.userId);
  if (failed.length > 0) return failed.filter((id) => ticked.has(id));
  return accounts
    .filter((account) => !account.banned && ticked.has(account.id))
    .map((account) => account.id);
}

export function hasFailedBans(results: Map<string, BanAccountResult>) {
  return [...results.values()].some((result) => !result.ok);
}

export function mergeBanResults(
  previous: Map<string, BanAccountResult>,
  next: BanAccountResult[],
) {
  const merged = new Map(previous);
  for (const result of next) merged.set(result.userId, result);
  const failed = [...merged.values()].filter((result) => !result.ok).length;
  return { merged, total: merged.size, failed };
}

export function partialBanFailure(
  total: number,
  failed: number,
  otherErrors: string[] = [],
) {
  return {
    title: `Banned ${total - failed} of ${pluralize(total, "account")}`,
    description: [`${failed} failed; the dialog shows which.`, ...otherErrors]
      .join(" ")
      .trim(),
    variant: "destructive" as const,
  };
}

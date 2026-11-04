import type { Ask } from "@/app/api/__generated__/models/ask";
import type { Criterion } from "@/app/api/__generated__/models/criterion";

/** Checks the work didn't meet, so accepting it is a choice made knowingly. */
export function unmetChecks(doneWhen: Criterion[] | undefined) {
  return (doneWhen ?? [])
    .filter((c) => !c.status || c.status === "unmet" || c.status === "pending")
    .map((c) => c.text);
}

/** The worker's pick first, so it is the primary button. */
export function orderOptions(ask: Ask) {
  const options = ask.options ?? [];
  const pick = ask.default_option;
  if (!pick || !options.includes(pick)) return options;
  return [pick, ...options.filter((o) => o !== pick)];
}

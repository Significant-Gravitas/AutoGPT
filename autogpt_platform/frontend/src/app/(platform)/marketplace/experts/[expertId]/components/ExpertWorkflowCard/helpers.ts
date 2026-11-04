import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";

// The humanizer falls back to "Scheduled" or "Cron Expression: …", neither of
// which reads after "Runs".
export function getCadenceLabel(cron: string) {
  const phrase = safeHumanizeCronExpression(cron);
  if (!/^(Every|On day)\b/.test(phrase)) return "Runs on a schedule";
  return `Runs ${phrase.charAt(0).toLowerCase()}${phrase.slice(1)}`;
}

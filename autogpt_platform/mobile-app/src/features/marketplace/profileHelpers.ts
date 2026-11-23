// Relative imports: jest has no `@/` alias.
import type { ExpertTemplate, ExpertWorkflow } from "../../lib/api/experts";

const CREDITS_PER_DOLLAR = 100;

function plural(count: number, word: string) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/** The hero's chips, as on the web: only the counts that are not zero. */
export function profileStats(
  template: Pick<ExpertTemplate, "bundled_skills" | "workflows">,
  serviceCount: number,
) {
  const skillCount = template.bundled_skills.length;
  const workflowCount = template.workflows?.length ?? 0;
  return [
    skillCount > 0 ? plural(skillCount, "skill") : null,
    workflowCount > 0 ? plural(workflowCount, "workflow") : null,
    serviceCount > 0 ? plural(serviceCount, "service") : null,
  ].filter((stat): stat is string => stat !== null);
}

/**
 * The services the expert's workflows sign in to, less the ones the platform
 * pays for. Empty until that list is in, so it never names a service you will
 * not have to connect.
 */
export function accessProviders(
  workflows: ExpertWorkflow[] | undefined,
  systemProviders: string[] | undefined,
) {
  if (!systemProviders) return [];
  const supplied = new Set(systemProviders);
  const providers = (workflows ?? [])
    .flatMap((workflow) => workflow.integration_providers ?? [])
    .filter((provider) => !supplied.has(provider));
  return [...new Set(providers)];
}

/** What comes with a hire, as the web's hire card lists it. */
export function hirePerks(workflowCount: number) {
  return [
    "No extra cost",
    "Joins your team instantly",
    ...(workflowCount > 0
      ? [`${plural(workflowCount, "workflow")} installed and ready to run`]
      : []),
    "Asks before acting on your behalf",
  ];
}

/** The web's line on what a hire costs, with the weekly cap when there is one. */
export function planLine(name: string, weeklyBudget: number | null | undefined) {
  const start = `Hiring ${name} costs nothing extra. Their work draws on your existing balance`;
  if (!weeklyBudget) return `${start}.`;
  return `${start}, capped at ${dollars(weeklyBudget)} a week — a limit you can change once they are on your team.`;
}

function dollars(credits: number) {
  const amount = credits / CREDITS_PER_DOLLAR;
  return Number.isInteger(amount) ? `$${amount.toLocaleString("en-US")}` : `$${amount.toFixed(2)}`;
}

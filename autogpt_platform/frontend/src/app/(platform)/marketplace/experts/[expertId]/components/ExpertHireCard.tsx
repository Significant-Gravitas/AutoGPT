import { Expert } from "@/app/api/__generated__/models/expert";
import { Icon } from "@/components/atoms/Icon/Icon";
import { creditsToUsdLabel } from "@/lib/credits";
import { CheckmarkCircle02Icon, Coins01Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";

interface Props {
  expert: Expert;
  firstName: string;
  actions: ReactNode;
}

/** The page's call to action, kept in view: sticky beside the content on
 *  wide screens, straight under the hero on phones. It carries the price
 *  terms so the decision and its cost sit together. */
export function ExpertHireCard({ expert, firstName, actions }: Props) {
  const weeklyBudget = expert.weekly_budget ?? null;
  const workflowCount = expert.workflows.length;

  return (
    <aside className="rounded-2xl bg-surface p-5 smooth-shadow-ring-sm">
      <div className="[&_a]:w-full [&_button]:w-full">{actions}</div>

      <ul className="mt-5 flex flex-col gap-2.5">
        <Perk icon={CheckmarkCircle02Icon}>No extra cost</Perk>
        <Perk icon={CheckmarkCircle02Icon}>Joins your team instantly</Perk>
        {workflowCount > 0 ? (
          <Perk icon={CheckmarkCircle02Icon}>
            {`${workflowCount} ${workflowCount === 1 ? "workflow" : "workflows"} installed and ready to run`}
          </Perk>
        ) : null}
        <Perk icon={CheckmarkCircle02Icon}>
          Asks before acting on your behalf
        </Perk>
      </ul>

      <div className="mt-5 border-t border-separator pt-4">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-foreground-strong">
          <Icon
            icon={Coins01Icon}
            size={16}
            className="text-foreground-tertiary"
          />
          Included with your plan
        </h2>
        <p className="mt-1.5 text-sm leading-6 text-foreground-secondary">
          {`Hiring ${firstName} costs nothing extra. Their work draws on your existing balance`}
          {weeklyBudget
            ? `, capped at ${creditsToUsdLabel(weeklyBudget)} a week — a limit you can change once they are on your team.`
            : "."}
        </p>
      </div>
    </aside>
  );
}

function Perk({
  icon,
  children,
}: {
  icon: typeof CheckmarkCircle02Icon;
  children: ReactNode;
}) {
  return (
    <li className="flex items-start gap-2 text-sm text-foreground-body">
      <Icon
        icon={icon}
        size={16}
        // TODO(tokens): no exact role: text-emerald-600 #059669
        className="mt-0.5 shrink-0 text-emerald-600"
        aria-hidden
      />
      <span>{children}</span>
    </li>
  );
}

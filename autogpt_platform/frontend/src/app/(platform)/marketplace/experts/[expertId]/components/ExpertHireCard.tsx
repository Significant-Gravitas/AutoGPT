import { Expert } from "@/app/api/__generated__/models/expert";
import { Icon } from "@/components/atoms/Icon/Icon";
import { creditsToUsdLabel } from "@/lib/credits";
import { getExpertTopicHex } from "@/components/molecules/ExpertAvatar/colors";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import {
  CheckmarkCircle02Icon,
  Coins01Icon,
  FlashIcon,
} from "@hugeicons/core-free-icons";
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
    <aside className="self-start rounded-2xl bg-white p-5 smooth-shadow-ring-sm lg:sticky lg:top-6">
      <div className="flex items-center gap-3">
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatar_url}
          color={expert.color}
          backgroundColor={getExpertTopicHex({
            avatarUrl: expert.avatar_url,
            categories: expert.categories,
            role: expert.role,
          })}
          size={44}
          className="rounded-full border border-[#e3e3e3]"
        />
        <div className="min-w-0">
          <p className="truncate text-base font-semibold text-zinc-900">
            Hire {firstName}
          </p>
          <p className="text-sm text-zinc-500">Joins your team in seconds</p>
        </div>
      </div>

      <div className="mt-4 [&_a]:w-full [&_button]:w-full">{actions}</div>

      <ul className="mt-5 flex flex-col gap-2.5">
        {workflowCount > 0 ? (
          <Perk icon={FlashIcon}>
            {`${workflowCount} ${workflowCount === 1 ? "workflow" : "workflows"} installed and ready to run`}
          </Perk>
        ) : null}
        <Perk icon={CheckmarkCircle02Icon}>
          Asks before acting on your behalf
        </Perk>
      </ul>

      <div className="mt-5 border-t border-zinc-100 pt-4">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-zinc-900">
          <Icon icon={Coins01Icon} size={16} className="text-zinc-500" />
          Included with your plan
        </h2>
        <p className="mt-1.5 text-sm leading-6 text-zinc-600">
          {`Hiring ${firstName} costs nothing extra. Their work draws on your existing balance`}
          {weeklyBudget
            ? `, capped at ${creditsToUsdLabel(weeklyBudget)} a week — a limit you can change once they are on your team.`
            : "."}
        </p>
      </div>

      <p className="mt-4 text-xs leading-5 text-zinc-500">
        Hire an AI Expert for supported tasks you might otherwise contract out.
        Review outputs before using them.
      </p>
    </aside>
  );
}

function Perk({
  icon,
  children,
}: {
  icon: typeof FlashIcon;
  children: ReactNode;
}) {
  return (
    <li className="flex items-start gap-2 text-sm text-zinc-700">
      <Icon
        icon={icon}
        size={16}
        className="mt-0.5 shrink-0 text-emerald-600"
        aria-hidden
      />
      <span>{children}</span>
    </li>
  );
}

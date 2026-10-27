import type { PlanResponse } from "@/app/api/__generated__/models/planResponse";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { useOnboardingWizardStore } from "../../../../store";
import {
  type ConsentMoment,
  getStepTasks,
  getTaskOwner,
  PLAN_STEP_TASKS,
  roleHeard,
} from "./helpers";
import { PlanConsentLine } from "./PlanConsentLine";
import { PlanTaskCard } from "./PlanTaskCard";

export const PLAN_HEADING_ID = "plan-heading";

interface Props {
  plan: PlanResponse | null;
  moment: ConsentMoment;
  /** The whole card (no paywall). */
  standalone?: boolean;
  /** The plan's heading is the page's title: standalone, or while the
   * trial's own heading is still loading. */
  isPageTitle?: boolean;
  footer?: ReactNode;
}

// Otto's plan: what he heard, the first three tasks and who takes each, and
// when they join. Nothing here is a control; the step's only action is the
// trial (or Continue) beside or below it.
export function PlanPanel({
  plan,
  moment,
  standalone = false,
  isPageTitle = standalone,
  footer,
}: Props) {
  return (
    <div
      data-testid="plan-panel"
      className={cn(
        "flex flex-col gap-3.5 bg-surface-muted px-5 pb-5 pt-6 sm:px-7 [@media(max-height:780px)]:gap-2.5 [@media(max-height:780px)]:py-5",
        !standalone && "border-b border-border-base lg:border-b-0 lg:border-r",
      )}
    >
      <div className="flex items-center gap-2">
        <AutopilotAvatar size={26} className="rounded-full" />
        <Text variant="small-medium" as="span" tone="secondary">
          Otto&apos;s plan
        </Text>
      </div>
      <Text
        variant="h4"
        as={isPageTitle ? "h1" : "h2"}
        id={PLAN_HEADING_ID}
        className="text-balance"
      >
        Here&apos;s what your team does first
      </Text>
      {plan ? <PlanBody plan={plan} moment={moment} /> : <PlanSkeleton />}
      {footer}
    </div>
  );
}

function PlanBody({
  plan,
  moment,
}: {
  plan: PlanResponse;
  moment: ConsentMoment;
}) {
  const role = useOnboardingWizardStore((s) => s.role);
  const team = plan.team ?? [];
  const heard = plan.source === "llm" ? plan.heard : null;
  return (
    <>
      <HeardCard
        label={heard ? "What I heard" : "Based on your role"}
        text={heard ?? roleHeard({ role, skippedDump: plan.skipped_dump })}
      />
      <ul aria-label="Your first tasks" className="flex flex-col gap-2">
        {getStepTasks(plan).map((task, index) => (
          <PlanTaskCard
            key={task.id}
            task={task}
            owner={getTaskOwner(task, team)}
            isFirst={index === 0}
          />
        ))}
      </ul>
      <PlanConsentLine team={team} moment={moment} />
    </>
  );
}

function HeardCard({ label, text }: { label: string; text: string }) {
  return (
    <div className="rounded-2xl border border-border-base bg-surface px-3.5 py-2.5">
      <Text variant="eyebrow" as="p" className="mb-1">
        {label}
      </Text>
      <Text variant="body" as="p" tone="secondary" className="text-pretty">
        {text}
      </Text>
    </div>
  );
}

// Sized like the plan it stands in for, so the card does not jump when the
// plan lands: the heard card, three task cards, then the consent line.
function PlanSkeleton() {
  return (
    <div
      data-testid="plan-panel-pending"
      className="flex flex-col gap-3.5 [@media(max-height:780px)]:gap-2.5"
      aria-busy
    >
      <span className="sr-only" role="status">
        Otto is writing your plan
      </span>
      <Skeleton className="h-[5.375rem] rounded-2xl" />
      <div className="flex flex-col gap-2">
        {Array.from({ length: PLAN_STEP_TASKS }, (_, index) => (
          <Skeleton key={index} className="h-[6.75rem] rounded-2xl" />
        ))}
      </div>
      <Skeleton className="h-8 w-4/5 rounded-lg" />
    </div>
  );
}

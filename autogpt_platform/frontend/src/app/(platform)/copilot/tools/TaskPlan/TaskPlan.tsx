"use client";

import type { ToolUIPart } from "ai";
import { Task01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatCost } from "@/app/(platform)/tasks/data/helpers";
import type { TaskPlan } from "@/app/(platform)/tasks/data/types";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolErrorCard } from "../../components/ToolErrorCard/ToolErrorCard";
import { ToolIcon } from "../DecomposeGoal/helpers";
import { PlanStartedCard } from "./components/PlanStartedCard";
import { PlanStepRow } from "./components/PlanStepRow";
import { getConfirmedPlan, getErrorMessage, getProposedPlan } from "./helpers";
import { useTaskPlanCard } from "./useTaskPlanCard";

interface Props {
  part: ToolUIPart;
}

export function ProposePlanTool({ part }: Props) {
  const plan = getProposedPlan(part.output);
  const errorMessage = getErrorMessage(part.output);
  const isError = part.state === "output-error" || !!errorMessage;

  if (plan) return <PlanCard plan={plan} />;
  if (isError) {
    return (
      <div className="py-2">
        <ToolErrorCard
          message={errorMessage ?? ""}
          fallbackMessage="Otto couldn't put the plan together. Please try again."
          actions={[]}
        />
      </div>
    );
  }
  return <PendingRow text="Drafting a plan…" />;
}

export function ConfirmTaskPlanTool({ part }: Props) {
  const confirmed = getConfirmedPlan(part.output);
  const errorMessage = getErrorMessage(part.output);

  if (confirmed) {
    return (
      <div className="py-2">
        <PlanStartedCard
          goal={confirmed.plan?.title}
          parentId={confirmed.parent?.id ?? confirmed.plan?.task_id ?? null}
          parentNumber={confirmed.parent?.number || null}
        />
      </div>
    );
  }
  if (part.state === "output-error" || errorMessage) {
    return (
      <div className="py-2">
        <ToolErrorCard
          message={errorMessage ?? ""}
          fallbackMessage="The plan couldn't be started. Please try again."
          actions={[]}
        />
      </div>
    );
  }
  return <PendingRow text="Starting the plan…" />;
}

function PendingRow({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-2 py-2">
      <ToolIcon isStreaming />
      <MorphingTextAnimation text={text} />
    </div>
  );
}

export function PlanCard({ plan }: { plan: TaskPlan }) {
  const card = useTaskPlanCard(plan);

  return (
    <div className="flex flex-col gap-2 py-2">
      <div className="flex flex-col gap-3 rounded-card bg-surface p-4 shadow-card">
        <div className="flex items-start gap-2.5">
          <span className="flex h-5 shrink-0 items-center">
            <Icon icon={Task01Icon} size={16} tone="quiet" aria-hidden />
          </span>
          <div className="flex min-w-0 flex-1 flex-col gap-0.5">
            <Text variant="inbox-medium" tone="quiet">
              Plan
            </Text>
            <Text variant="inbox-medium" tone="ink" className="text-pretty">
              {plan.title}
            </Text>
            {plan.summary ? (
              <Text variant="inbox" tone="text" className="text-pretty">
                {plan.summary}
              </Text>
            ) : null}
          </div>
        </div>

        <ol className="divide-y divide-border-hairline border-y border-border-hairline">
          {plan.steps.map((step, i) => (
            <PlanStepRow key={`${i}-${step.title}`} step={step} index={i} />
          ))}
        </ol>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <Text variant="inbox" tone="quiet" tabular>
            {plan.steps.length} {plan.steps.length === 1 ? "step" : "steps"}
            {plan.budget_usd !== null
              ? ` · Budget ${formatCost(plan.budget_usd)}`
              : ""}
          </Text>
          {!card.isStarted && !card.isCancelled ? (
            <Button
              variant="primary"
              size="small"
              loading={card.isApproving}
              disabled={card.isApproving}
              onClick={card.handleApprove}
            >
              Approve & start
            </Button>
          ) : null}
          {card.isCancelled ? (
            <Text variant="inbox-medium" tone="quiet">
              Plan cancelled
            </Text>
          ) : null}
        </div>
      </div>
      {card.isStarted ? (
        <PlanStartedCard
          parentId={card.parentId}
          parentNumber={card.parentNumber}
        />
      ) : null}
    </div>
  );
}

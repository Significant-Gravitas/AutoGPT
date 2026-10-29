"use client";

import {
  ContentGrid,
  ContentMessage,
} from "@/app/(platform)/copilot/components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "@/app/(platform)/copilot/components/ToolAccordion/ToolAccordion";
import { Text } from "@/components/atoms/Text/Text";
import type { CSSProperties } from "react";
import type { TourPlan, TourPlanStep } from "../../script/types";
import {
  CheckmarkCircle02Icon,
  CubeIcon,
  PlusSignCircleIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

const STEP_STAGGER_MS = 200;

/** Tour-local twin of the copilot's DecomposeGoal card — same ToolAccordion
 * shell, but the steps stream in with a staggered fade/slide and the check
 * icons pop in after each row. */
export function TourPlanCard({ plan }: { plan: TourPlan }) {
  return (
    <ToolAccordion
      icon={<Icon icon={PlusSignCircleIcon} size={32} />}
      title={`Build Plan — ${plan.steps.length} steps`}
      description={plan.goal}
      defaultExpanded
    >
      <ContentGrid>
        <ContentMessage>
          {`Here's the plan (${plan.steps.length} steps):`}
        </ContentMessage>

        <div className="mb-6 rounded-lg bg-card p-3">
          <div className="space-y-0.5">
            {plan.steps.map((step, index) => (
              <TourPlanStepItem
                key={`${step.blockName}-${index}`}
                step={step}
                index={index}
              />
            ))}
          </div>
        </div>
      </ContentGrid>
    </ToolAccordion>
  );
}

function TourPlanStepItem({
  step,
  index,
}: {
  step: TourPlanStep;
  index: number;
}) {
  return (
    <div
      // oxlint-disable-next-line design-system/no-arbitrary-values -- per-row stagger delay set at runtime through --step-delay
      className="flex items-start gap-3 py-1.5 duration-500 animate-in fade-in slide-in-from-bottom-2 fill-mode-both [animation-delay:var(--step-delay)]"
      style={
        {
          "--step-delay": `${index * STEP_STAGGER_MS}ms`,
          "--check-delay": `${index * STEP_STAGGER_MS + 200}ms`,
        } as CSSProperties
      }
    >
      {/* oxlint-disable-next-line design-system/no-arbitrary-values -- per-row check pop-in delay set at runtime through --check-delay */}
      <span className="mt-0.5 flex shrink-0 text-success duration-300 animate-in fade-in zoom-in-50 fill-mode-both [animation-delay:var(--check-delay)]">
        <Icon icon={CheckmarkCircle02Icon} size={18} aria-label="completed" />
      </span>
      <div className="min-w-0 flex-1">
        <Text variant="large-medium-tight">
          {index + 1}. {step.description}
        </Text>
        <div className="mt-0.5 flex items-center gap-1 text-foreground-tertiary">
          <Icon icon={CubeIcon} size={12} />
          <Text variant="small-mono-tight" as="p" tone="muted">
            {step.blockName}
          </Text>
        </div>
      </div>
    </div>
  );
}

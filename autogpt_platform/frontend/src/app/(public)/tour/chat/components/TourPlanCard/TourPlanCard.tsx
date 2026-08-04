"use client";

import {
  ContentGrid,
  ContentMessage,
} from "@/app/(platform)/copilot/components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "@/app/(platform)/copilot/components/ToolAccordion/ToolAccordion";
import { Text } from "@/components/atoms/Text/Text";
import {
  CheckCircleIcon,
  CubeIcon,
  PlusCircleIcon,
} from "@phosphor-icons/react";
import type { TourPlan, TourPlanStep } from "../../script/types";

const STEP_STAGGER_MS = 200;

/** Tour-local twin of the copilot's DecomposeGoal card — same ToolAccordion
 * shell, but the steps stream in with a staggered fade/slide and the check
 * icons pop in after each row. */
export function TourPlanCard({ plan }: { plan: TourPlan }) {
  return (
    <ToolAccordion
      icon={<PlusCircleIcon size={32} weight="light" />}
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
      className="flex items-start gap-3 py-1.5 duration-500 animate-in fade-in slide-in-from-bottom-2 fill-mode-both"
      style={{ animationDelay: `${index * STEP_STAGGER_MS}ms` }}
    >
      <CheckCircleIcon
        size={18}
        weight="fill"
        aria-label="completed"
        className="mt-0.5 shrink-0 text-emerald-500 duration-300 animate-in fade-in zoom-in-50 fill-mode-both"
        style={{ animationDelay: `${index * STEP_STAGGER_MS + 200}ms` }}
      />
      <div className="min-w-0 flex-1">
        <Text variant="body-medium" className="text-base text-foreground">
          {index + 1}. {step.description}
        </Text>
        <div className="mt-0.5 flex items-center gap-1">
          <CubeIcon size={12} className="text-muted-foreground" />
          <Text
            variant="small"
            className="font-mono text-xs text-muted-foreground"
          >
            {step.blockName}
          </Text>
        </div>
      </div>
    </div>
  );
}

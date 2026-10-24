"use client";

import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { cn } from "@/lib/utils";
import { useRef } from "react";
import { HireStepActions } from "./components/HireStepActions";
import {
  EXPERT_CARD_CLASS,
  RecommendedExpertCard,
} from "./components/RecommendedExpertCard";
import { hireDiagnosis, hireTitle, raiseNote } from "./helpers";
import { useHireStep } from "./useHireStep";
import { useScrollEdges } from "./useScrollEdges";

const SKELETON_CARDS = 3;
// Three across at the container's max width, two on tablets, one on phones.
const CARD_WIDTH_CLASS = "w-full sm:w-[calc(50%-0.375rem)] lg:w-72";
const EDGE_FADE_CLASS =
  // TODO(tokens): no exact role for from-gray-100 (#F3F4F6), via-gray-100/70 (#F3F4F6)
  "pointer-events-none absolute inset-x-0 z-10 h-16 from-gray-100 via-gray-100/70 to-transparent transition-opacity duration-200";

// Otto's first act as Head of AI: right after the brain dump, before the
// user is asked to pay for anything, it proposes the experts that take the
// problems it heard off the user's plate. One click hires the whole team, or
// one card at a time; the voice pick waits for the expert's own page.
export function HireStep() {
  const step = useHireStep();
  const note = raiseNote(step.raiseRole);
  const diagnosis = hireDiagnosis(step.team, step.isPending);
  const isBusy = step.isCardHiring || step.isHiringAll;
  const gridRef = useRef<HTMLDivElement>(null);
  const { hiddenAbove, hiddenBelow } = useScrollEdges(gridRef);

  return (
    <FadeIn>
      <div className="flex w-full flex-col items-center gap-8 px-4">
        <div className="mx-auto flex w-full max-w-lg flex-col items-center gap-4 px-4 text-center">
          <AutopilotAvatar size={120} transparent />
          <Text variant="h4" as="h1">
            {hireTitle(step.team, step.isPending)}
          </Text>
          {diagnosis ? (
            <Text
              variant="body"
              tone="muted"
              className="max-w-lg text-center"
              data-testid="hire-step-diagnosis"
            >
              {diagnosis}
            </Text>
          ) : null}
        </div>

        {/* The roster can run to sixteen cards: the row wraps and scrolls
            inside a fixed height with no scrollbar, and a fade appears only on
            the edge that actually has cards hidden behind it. Wrapping flex
            rather than a grid so a short last row sits centered under the
            full ones instead of hugging the left. */}
        <div className="relative w-full max-w-4xl">
          <div
            aria-hidden
            className={cn(
              EDGE_FADE_CLASS,
              "top-0 bg-gradient-to-b",
              hiddenAbove ? "opacity-100" : "opacity-0",
            )}
          />
          <div
            aria-hidden
            className={cn(
              EDGE_FADE_CLASS,
              "bottom-0 bg-gradient-to-t",
              hiddenBelow ? "opacity-100" : "opacity-0",
            )}
          />
          <div
            ref={gridRef}
            className="flex max-h-[26rem] w-full flex-wrap justify-center gap-3 overflow-y-auto scrollbar-none"
            data-testid="hire-step-grid"
          >
            {step.isPending
              ? Array.from({ length: SKELETON_CARDS }, (_, i) => (
                  <div
                    key={i}
                    className={cn(
                      EXPERT_CARD_CLASS,
                      CARD_WIDTH_CLASS,
                      "border-border-base",
                    )}
                    data-testid="hire-step-pending"
                  >
                    <div className="flex items-center gap-3">
                      <Skeleton className="size-10 rounded-full" />
                      <div className="flex flex-col gap-2">
                        <Skeleton className="h-3 w-24" />
                        <Skeleton className="h-3 w-16" />
                      </div>
                    </div>
                    <Skeleton className="h-3 w-56" />
                    <Skeleton className="h-7 w-14 rounded-full" />
                  </div>
                ))
              : step.experts.map((expert, index) => (
                  <div key={expert.template_id} className={CARD_WIDTH_CLASS}>
                    <RecommendedExpertCard
                      expert={expert}
                      isHired={step.hiredTemplateIds.includes(
                        expert.template_id,
                      )}
                      isHiringThis={step.isHiringCard(expert.template_id)}
                      isIncluded={step.isIncluded(expert.template_id)}
                      disabled={isBusy}
                      onHire={() => step.hire(expert, index)}
                      onToggleIncluded={() =>
                        step.toggleIncluded(expert.template_id)
                      }
                    />
                  </div>
                ))}
          </div>
        </div>

        {note ? (
          <Text
            variant="small"
            tone="muted"
            className="max-w-lg text-center"
            data-testid="hire-step-raise-note"
          >
            {note}
          </Text>
        ) : null}

        <HireStepActions
          isPending={step.isPending}
          isHiringAll={step.isHiringAll}
          isCardHiring={step.isCardHiring}
          primaryLabel={step.primaryLabel}
          secondaryLabel={step.secondaryLabel}
          onPrimary={step.handlePrimary}
          onSecondary={step.handleSecondary}
        />
      </div>
    </FadeIn>
  );
}

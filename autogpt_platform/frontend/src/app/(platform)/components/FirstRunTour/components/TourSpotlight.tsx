"use client";

import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "@/components/molecules/Popover/Popover";
import { useStableCallback } from "@/hooks/useStableCallback";
import { useEffect, useId, useRef } from "react";
import type { Box } from "../helpers";
import { stepListOf, type TourTarget } from "../steps";
import type { useFirstRunTour } from "../useFirstRunTour";
import { useTourStops } from "../useTourStops";
import { TourScrim } from "./TourScrim";
import { TourStepCard } from "./TourStepCard";

interface Props {
  tour: ReturnType<typeof useFirstRunTour>;
}

// Sidebar targets sit on the left edge, so the card opens to their right; the
// phone's menu button sits top-left, so the card drops below it; the first
// tasks sit mid-page, so the card sits above them and flips if it must.
function sideFor(target: TourTarget | null) {
  if (target === "menu") return "bottom";
  if (target === "first-tasks" || target === "composer") return "top";
  if (target === null) return "bottom";
  return "right";
}

function anchorRect(box: Box | null) {
  if (box) return DOMRect.fromRect({ ...box, x: box.left, y: box.top });
  // Nothing to point at (the first tasks are still on their way): hold the
  // card in the middle of the screen until the target arrives.
  return DOMRect.fromRect({
    x: window.innerWidth / 2,
    y: window.innerHeight / 3,
    width: 0,
    height: 0,
  });
}

// One surface at a time: the app dims, a window opens over the real control,
// and a card beside it says what it is in one line.
export function TourSpotlight({ tour }: Props) {
  const step = tour.steps[tour.stepIndex];
  const [measured] = useTourStops(stepListOf(step));
  const stop = measured?.stepID === step.id ? measured : undefined;
  const box = stop?.box ?? null;
  // Read on every frame by the card's positioning, so it always sees the
  // latest box without the anchor object changing identity.
  const boxRef = useRef(box);
  useEffect(() => {
    boxRef.current = box;
  }, [box]);
  const virtualRef = useRef({
    getBoundingClientRect: () => anchorRect(boxRef.current),
  });
  const contentRef = useRef<HTMLDivElement>(null);
  const titleID = useId();
  const bodyID = useId();

  const body = tour.bodyFor(tour.stepIndex, stop);
  const target = stop?.target ?? null;

  // Once per step, as soon as it knows what it points at.
  const trackedStepRef = useRef<string | null>(null);
  const trackStep = useStableCallback(tour.trackStep);
  useEffect(() => {
    if (!stop || trackedStepRef.current === step.id) return;
    trackedStepRef.current = step.id;
    trackStep(stop);
  }, [stop, step.id, trackStep]);

  function focusPrimary() {
    contentRef.current
      ?.querySelector<HTMLElement>("[data-tour-primary]")
      ?.focus();
  }

  useEffect(focusPrimary, [step.id]);

  return (
    <>
      <TourScrim boxes={box ? [box] : []} />
      <Popover open modal>
        <PopoverAnchor virtualRef={virtualRef} />
        <PopoverContent
          side={sideFor(target)}
          align={target === null ? "center" : "start"}
          sideOffset={14}
          collisionPadding={16}
          updatePositionStrategy="always"
          ref={contentRef}
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            focusPrimary();
          }}
          onEscapeKeyDown={(event) => {
            event.preventDefault();
            tour.skip();
          }}
          onInteractOutside={(event) => event.preventDefault()}
          aria-labelledby={titleID}
          aria-describedby={bodyID}
          data-testid="first-run-tour"
          className="z-[120] w-[21rem] max-w-[calc(100vw-2rem)] rounded-2xl border-border-base p-5 shadow-xl"
        >
          <TourStepCard
            icon={step.icon}
            title={step.title}
            body={body}
            titleID={titleID}
            bodyID={bodyID}
            stepNumber={tour.stepIndex + 1}
            stepCount={tour.steps.length}
            viaMenu={stop?.viaMenu ?? false}
            isFirstStep={tour.stepIndex === 0}
            isLastStep={tour.isLastStep}
            onBack={tour.back}
            onNext={tour.next}
            onSkip={tour.skip}
            altAction={
              tour.isLastStep
                ? {
                    label: "Or connect your tools first",
                    onClick: tour.connectToolsFirst,
                  }
                : undefined
            }
          />
          {/* Inside the card: a modal popover hides everything outside it
              from assistive tech, and focus stays on the same button from
              step to step, so this is what announces each new step. */}
          <p className="sr-only" aria-live="polite">
            {`Step ${tour.stepIndex + 1} of ${tour.steps.length}. ${step.title}. ${body}`}
          </p>
        </PopoverContent>
      </Popover>
    </>
  );
}

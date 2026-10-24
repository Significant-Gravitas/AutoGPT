"use client";

import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "@/components/molecules/Popover/Popover";
import { useSidebar } from "@/components/ui/sidebar";
import { useStableCallback } from "@/hooks/useStableCallback";
import { useEffect, useId, useRef, useState } from "react";
import { isPartlyOffScreen, withMenuHint, type Box } from "../helpers";
import type { TourTarget } from "@/lib/tour-targets";
import { stepListOf } from "../steps";
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

  const { isMobile, openMobile } = useSidebar();
  const isOverSheet = isMobile && openMobile;
  const layer = useLayerAboveSheet(isOverSheet);
  const body = withMenuHint(
    tour.bodyFor(tour.stepIndex, stop),
    stop?.viaMenu ?? false,
  );
  const target = stop?.target ?? null;
  const element = stop?.element ?? null;

  useEffect(() => {
    if (!element || !isPartlyOffScreen(element, target)) return;
    element.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [element, target]);

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
      <Popover open modal key={layer}>
        <PopoverAnchor virtualRef={virtualRef} />
        <PopoverContent
          // Over the phone's sheet there is no room beside it: below the
          // item (above it, near the bottom) keeps the card on screen.
          side={isOverSheet ? "bottom" : sideFor(target)}
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

// The phone's sidebar sheet is a modal dialog too: one opened after the card
// takes its focus and taps and hides it from assistive tech. So the card
// mounts again once the sheet has opened (or closed), a couple of frames on,
// after the sheet's own effects have run in the same commit as its content.
function useLayerAboveSheet(isOverSheet: boolean) {
  const [layer, setLayer] = useState(isOverSheet ? "over-sheet" : "over-page");
  useEffect(() => {
    let frame = requestAnimationFrame(() => {
      frame = requestAnimationFrame(() =>
        setLayer(isOverSheet ? "over-sheet" : "over-page"),
      );
    });
    return () => cancelAnimationFrame(frame);
  }, [isOverSheet]);
  return layer;
}

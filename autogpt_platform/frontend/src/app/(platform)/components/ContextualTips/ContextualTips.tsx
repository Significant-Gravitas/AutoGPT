"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "@/components/molecules/Popover/Popover";
import { useStableCallback } from "@/hooks/useStableCallback";
import { trackFirstRun } from "@/services/onboarding/first-run-analytics";
import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";
import { useEffect, useId, useRef } from "react";
import type { Box } from "../FirstRunTour/helpers";
import { TourScrim } from "../FirstRunTour/components/TourScrim";
import { useTourStops } from "../FirstRunTour/useTourStops";
import { tipStepOf } from "./helpers";
import { getTipBody, TIPS, type TipID } from "./tips";
import { useContextualTips } from "./useContextualTips";

// Rendered once in the app chrome; shows at most one tip at a time.
export function ContextualTips() {
  const { active, dismiss, postponeActive } = useContextualTips();
  if (!active) return null;
  return (
    <ContextualTip
      key={active}
      id={active}
      onDismiss={dismiss}
      onUnavailable={postponeActive}
    />
  );
}

// How long a tip waits for its target (a sidebar sliding in, a page still
// loading) before stepping aside for the next one in the queue.
export const TIP_TARGET_GRACE_MS = 1500;

interface Props {
  id: TipID;
  onDismiss: () => void;
  onUnavailable: () => void;
}

// A small card beside the real control, with a ring around it and no dim:
// the user is in the middle of something, so it never takes focus and never
// blocks the page. "Got it" or Escape puts it away for good.
function ContextualTip({ id, onDismiss, onUnavailable }: Props) {
  const tip = TIPS[id];
  const [stop] = useTourStops(tipStepOf(id));
  const box = stop?.box ?? null;
  const boxRef = useRef<Box | null>(box);
  useEffect(() => {
    boxRef.current = box;
  }, [box]);
  const virtualRef = useRef({
    getBoundingClientRect: () =>
      boxRef.current
        ? DOMRect.fromRect({
            x: boxRef.current.left,
            y: boxRef.current.top,
            width: boxRef.current.width,
            height: boxRef.current.height,
          })
        : DOMRect.fromRect({ x: -9999, y: -9999, width: 0, height: 0 }),
  });
  const { experience } = useUsageExperience();
  const titleID = useId();
  const body = getTipBody(id, { isTrial: experience.isActiveTrial });
  const isMenu = stop?.target === "menu";
  const hasBox = box !== null;
  const giveWay = useStableCallback(onUnavailable);

  // Counted when the card is on screen beside its target, not when it
  // takes its turn: a tip still waiting for its target was not seen.
  const wasShown = useRef(false);
  useEffect(() => {
    if (!hasBox || wasShown.current) return;
    wasShown.current = true;
    trackFirstRun("contextual_tip_shown", { tip: id });
  }, [hasBox, id]);

  useEffect(() => {
    if (hasBox) return;
    const timer = window.setTimeout(giveWay, TIP_TARGET_GRACE_MS);
    return () => window.clearTimeout(timer);
  }, [hasBox, giveWay]);

  // Nothing on screen to point at yet: wait rather than float in a corner.
  if (!box) return null;

  return (
    <>
      <TourScrim boxes={[box]} dim={false} />
      <Popover open modal={false}>
        <PopoverAnchor virtualRef={virtualRef} />
        <PopoverContent
          side={isMenu ? "bottom" : "right"}
          align="start"
          sideOffset={12}
          collisionPadding={16}
          updatePositionStrategy="always"
          onOpenAutoFocus={(event) => event.preventDefault()}
          onCloseAutoFocus={(event) => event.preventDefault()}
          onInteractOutside={(event) => event.preventDefault()}
          onEscapeKeyDown={onDismiss}
          role="region"
          aria-labelledby={titleID}
          data-testid="contextual-tip"
          className="z-[120] w-[18rem] max-w-[calc(100vw-2rem)] rounded-2xl border-border-base p-4 shadow-xl"
        >
          <div className="flex flex-col gap-2">
            <Text variant="small-medium" className="text-violet-700">
              Tip
            </Text>
            <Text
              as="h2"
              id={titleID}
              variant="body-medium"
              tone="primary"
              className="[overflow-wrap:anywhere]"
            >
              {tip.title}
            </Text>
            <Text
              variant="small"
              tone="secondary"
              className="text-pretty [overflow-wrap:anywhere]"
            >
              {body}
              {isMenu && " Open the menu to find it."}
            </Text>
            <Button
              type="button"
              variant="secondary"
              size="xs"
              onClick={onDismiss}
              className="mt-1 self-end"
            >
              Got it
            </Button>
          </div>
        </PopoverContent>
      </Popover>
      <p className="sr-only" role="status" aria-live="polite">
        {`Tip: ${tip.title}. ${body}`}
      </p>
    </>
  );
}

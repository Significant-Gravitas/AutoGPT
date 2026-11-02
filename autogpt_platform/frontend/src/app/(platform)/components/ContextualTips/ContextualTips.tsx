"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { BulbIcon } from "@hugeicons/core-free-icons";
import { AnchoredPanel } from "@/components/molecules/AnchoredPanel/AnchoredPanel";
import { isKey } from "@/lib/keyboard";
import { useStableCallback } from "@/hooks/useStableCallback";
import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";
import { useEffect, useId, useRef } from "react";
import { isOnScreen, withMenuHint, type Box } from "../FirstRunTour/helpers";
import { TourScrim } from "../FirstRunTour/components/TourScrim";
import { useTourStops } from "../FirstRunTour/useTourStops";
import { tipStepOf } from "./helpers";
import { getTipBody, TIPS, type TipID } from "./tips";
import { useContextualTips } from "./useContextualTips";

export function ContextualTips() {
  const { active, dismiss, reportShown, postponeActive, othersWaiting } =
    useContextualTips();
  if (!active) return null;
  return (
    <ContextualTip
      key={active}
      id={active}
      onDismiss={dismiss}
      onShown={reportShown}
      onUnavailable={postponeActive}
      othersWaiting={othersWaiting}
    />
  );
}

// How long a tip waits for its target before letting the next one go first.
export const TIP_TARGET_GRACE_MS = 1500;

interface Props {
  id: TipID;
  onDismiss: () => void;
  onShown: (id: TipID) => void;
  onUnavailable: () => void;
  /** Another tip is queued behind this one. */
  othersWaiting: boolean;
}

// No dim and no focus: the user is in the middle of something.
function ContextualTip({
  id,
  onDismiss,
  onShown,
  onUnavailable,
  othersWaiting,
}: Props) {
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
  const isMenu = stop?.target === "menu";
  const body = withMenuHint(
    getTipBody(id, { isTrial: experience.isActiveTrial }),
    isMenu,
  );
  const hasBox = isOnScreen(box);
  const giveWay = useStableCallback(onUnavailable);

  const reportOnScreen = useStableCallback(onShown);

  useEffect(() => {
    if (hasBox) reportOnScreen(id);
  }, [hasBox, id, reportOnScreen]);

  // Re-armed whenever another tip queues up behind this one.
  useEffect(() => {
    if (hasBox || !othersWaiting) return;
    const timer = window.setTimeout(giveWay, TIP_TARGET_GRACE_MS);
    return () => window.clearTimeout(timer);
  }, [hasBox, othersWaiting, giveWay]);
  const contentRef = useRef<HTMLDivElement>(null);

  if (!isOnScreen(box)) return null;

  return (
    <>
      <TourScrim boxes={[box]} dim={false} />
      <AnchoredPanel
        virtualRef={virtualRef}
        side={isMenu ? "bottom" : "right"}
        align="start"
        sideOffset={12}
        collisionPadding={16}
        updatePositionStrategy="always"
        // Escape from inside puts it away; one elsewhere is left alone.
        onKeyDown={(event) => {
          if (isKey(event, "Escape")) onDismiss();
        }}
        ref={contentRef}
        role="region"
        aria-labelledby={titleID}
        data-testid="contextual-tip"
        className="z-[120] w-[18rem] max-w-[calc(100vw-2rem)] rounded-2xl border-border-base p-4 shadow-xl"
      >
        <div className="flex flex-col gap-2">
          <span className="flex items-center gap-1.5 text-foreground-secondary">
            <Icon icon={BulbIcon} size={14} aria-hidden />
            <Text variant="small-medium" tone="inherit">
              Tip
            </Text>
          </span>
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
      </AnchoredPanel>
      <p className="sr-only" role="status" aria-live="polite">
        {`Tip: ${tip.title}. ${body}`}
      </p>
    </>
  );
}

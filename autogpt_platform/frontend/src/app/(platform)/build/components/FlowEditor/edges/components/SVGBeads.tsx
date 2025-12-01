// This component uses SVG animation [Will see in future if we can make it work]
// Problem - it doesn't work with real time updates

import { useEffect, useMemo, useRef, useState } from "react";
import { getLengthOfPathInPixels } from "../helpers";

const BEAD_SPACING = 12;
const BASE_STOP_DISTANCE = 15;
const ANIMATION_DURATION = 0.5;
const ANIMATION_DELAY_PER_BEAD = 0.05;

interface BeadsProps {
  beadUp: number;
  beadDown: number;
  edgePath: string;
  beadsKey: string;
}

export const SVGBeads = ({
  beadUp,
  beadDown,
  edgePath,
  beadsKey,
}: BeadsProps) => {
  const [removedBeads, setRemovedBeads] = useState<Set<number>>(new Set());
  const animateRef = useRef<SVGAElement | null>(null);

  const visibleBeads = useMemo(() => {
    return Array.from({ length: Math.max(0, beadUp) }, (_, i) => i).filter(
      (index) => !removedBeads.has(index),
    );
  }, [beadUp, removedBeads]);

  const totalLength = getLengthOfPathInPixels(edgePath);

  useEffect(() => {
    setRemovedBeads(new Set());
  }, [beadUp]);

  useEffect(() => {
    const elem = animateRef.current;
    if (elem) {
      const handleEnd = () => {
        if (beadDown > 0) {
          const beadsToRemove = Array.from(
            { length: beadDown },
            (_, i) => beadUp - beadDown + i,
          );

          beadsToRemove.forEach((beadIndex) => {
            setRemovedBeads((prev) => new Set(prev).add(beadIndex));
          });
        }
      };
      elem.addEventListener("endEvent", handleEnd);
      return () => elem.removeEventListener("endEvent", handleEnd);
    }
  }, [beadUp, beadDown]);

  return (
    <>
      {visibleBeads.map((index) => {
        const stopDistance = BASE_STOP_DISTANCE + index * BEAD_SPACING;
        const beadStopPoint =
          Math.max(0, totalLength - stopDistance) / totalLength;

        return (
          <circle key={`${beadsKey}-${index}`} r="5" fill="#8d8d95">
            <animateMotion
              ref={animateRef}
              dur={`${ANIMATION_DURATION}s`}
              repeatCount="1"
              fill="freeze"
              path={edgePath}
              begin={`${index * ANIMATION_DELAY_PER_BEAD}s`}
              keyPoints={`0;${beadStopPoint}`}
              keyTimes="0;1"
              calcMode="linear"
            />
          </circle>
        );
      })}
    </>
  );
};

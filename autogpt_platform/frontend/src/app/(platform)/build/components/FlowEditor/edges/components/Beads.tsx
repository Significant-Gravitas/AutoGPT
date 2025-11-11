import { useMemo } from "react";
import { getLengthOfPathInPixels } from "../helpers";

const BEAD_SPACING = 12;
const BASE_STOP_DISTANCE = 15;

interface BeadsProps {
  beadUp: number;
  beadDown: number;
  edgePath: string;
  beadsKey: string;
}

export const Beads = ({ beadUp, beadDown, edgePath, beadsKey }: BeadsProps) => {
  const visibleBeads = useMemo(() => {
    const count = beadUp - beadDown;
    return Array.from({ length: Math.max(0, count) }, (_, i) => i);
  }, [beadUp, beadDown]);

  const totalLength = getLengthOfPathInPixels(edgePath);

  return (
    <>
      {visibleBeads.map((index) => {
        const stopDistance = BASE_STOP_DISTANCE + index * BEAD_SPACING;
        const beadStopPoint =
          Math.max(0, totalLength - stopDistance) / totalLength;

        return (
          <circle key={`${beadsKey}-${index}`} r="5" fill="#8d8d95">
            <animateMotion
              dur="0.5s"
              repeatCount="1"
              fill="freeze"
              path={edgePath}
              begin={`${index * 0.3}s`}
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

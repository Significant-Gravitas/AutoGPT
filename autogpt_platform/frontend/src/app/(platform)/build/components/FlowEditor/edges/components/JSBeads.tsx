// This component uses JS animation
// Problem - It lags at real time updates, because of state change

import { useCallback, useEffect, useRef, useState } from "react";
import { getLengthOfPathInPixels } from "../helpers";

const BEAD_DIAMETER = 10;
const ANIMATION_DURATION = 500;
const DELTA_TIME = 16;

interface Bead {
  t: number;
  targetT: number;
  startTime: number;
}

interface BeadsProps {
  beadUp: number;
  beadDown: number;
  edgePath: string;
  beadsKey: string;
  isStatic?: boolean;
}

export const JSBeads = ({
  beadUp,
  beadDown,
  edgePath,
  beadsKey,
  isStatic = false,
}: BeadsProps) => {
  const [beads, setBeads] = useState<{
    beads: Bead[];
    created: number;
    destroyed: number;
  }>({ beads: [], created: 0, destroyed: 0 });

  const beadsRef = useRef(beads);
  const totalLength = getLengthOfPathInPixels(edgePath);

  const pathRef = useRef<SVGPathElement | null>(null);

  const getPointAtT = (t: number) => {
    if (!pathRef.current) {
      pathRef.current = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "path",
      );
    }
    pathRef.current.setAttribute("d", edgePath);
    const length = pathRef.current.getTotalLength();
    const point = pathRef.current.getPointAtLength(t * length);
    return { x: point.x, y: point.y };
  };

  const getTForDistance = (distanceFromEnd: number) => {
    const distance = Math.max(0, totalLength - distanceFromEnd);
    return Math.max(0, Math.min(1, distance / totalLength));
  };

  const setTargetPositions = useCallback(
    (beads: Bead[]) => {
      const distanceBetween = Math.min(
        (totalLength - BEAD_DIAMETER) / (beads.length + 1),
        BEAD_DIAMETER,
      );

      return beads.map((bead, index) => {
        const distanceFromEnd = BEAD_DIAMETER * 1.35;
        const targetPosition = distanceBetween * index + distanceFromEnd;
        const t = getTForDistance(targetPosition);

        return {
          ...bead,
          t: isStatic ? t : bead.t,
          targetT: t,
        };
      });
    },
    [totalLength, isStatic],
  );

  beadsRef.current = beads;

  useEffect(() => {
    pathRef.current = null;
  }, [edgePath]);

  useEffect(() => {
    if (
      beadUp === 0 &&
      beadDown === 0 &&
      (beads.created > 0 || beads.destroyed > 0)
    ) {
      setBeads({ beads: [], created: 0, destroyed: 0 });
      return;
    }

    if (beadUp > beads.created) {
      setBeads(({ beads, created, destroyed }) => {
        const newBeads = [];
        for (let i = 0; i < beadUp - created; i++) {
          newBeads.push({ t: 0, targetT: 0, startTime: Date.now() });
        }

        const b = setTargetPositions([...beads, ...newBeads]);
        return { beads: b, created: beadUp, destroyed };
      });
    }

    const interval = setInterval(
      ({ current: beads }) => {
        if (
          (beadUp === beads.created && beads.created === beads.destroyed) ||
          beads.beads.every((bead) => bead.t >= bead.targetT)
        ) {
          clearInterval(interval);
          return;
        }

        setBeads(({ beads, created, destroyed }) => {
          let destroyedCount = 0;

          const newBeads = beads
            .map((bead) => {
              const progressIncrement = DELTA_TIME / ANIMATION_DURATION;
              const t = Math.min(
                bead.t + bead.targetT * progressIncrement,
                bead.targetT,
              );

              return { ...bead, t };
            })
            .filter((bead, index) => {
              const removeCount = beadDown - destroyed;
              if (bead.t >= bead.targetT && index < removeCount) {
                destroyedCount++;
                return false;
              }
              return true;
            });

          return {
            beads: setTargetPositions(newBeads),
            created,
            destroyed: destroyed + destroyedCount,
          };
        });
      },
      DELTA_TIME,
      beadsRef,
    );

    return () => clearInterval(interval);
  }, [beadUp, beadDown, setTargetPositions, isStatic]);

  return (
    <>
      {beads.beads.map((bead, index) => {
        const pos = getPointAtT(bead.t);
        return (
          <circle
            key={`${beadsKey}-${index}`}
            cx={pos.x}
            cy={pos.y}
            r={BEAD_DIAMETER / 2}
            fill="#8d8d95"
          />
        );
      })}
    </>
  );
};

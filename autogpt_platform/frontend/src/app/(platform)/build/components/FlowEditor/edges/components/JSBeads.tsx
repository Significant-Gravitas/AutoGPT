// This component uses JS animation [It's replica of legacy builder]
// Problem - It lags at real time updates, because of state change

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getLengthOfPathInPixels,
  getPointAtT,
  getTForDistance,
  setTargetPositions,
} from "../helpers";

const BEAD_DIAMETER = 10;
const ANIMATION_DURATION = 500;

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
}: BeadsProps) => {
  const [beads, setBeads] = useState<{
    beads: Bead[];
    created: number;
    destroyed: number;
  }>({ beads: [], created: 0, destroyed: 0 });

  const beadsRef = useRef(beads);
  const totalLength = getLengthOfPathInPixels(edgePath);
  const animationFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);

  const pathRef = useRef<SVGPathElement | null>(null);

  const getPointAtTWrapper = (t: number) => {
    return getPointAtT(t, edgePath, pathRef);
  };

  const getTForDistanceWrapper = (distanceFromEnd: number) => {
    return getTForDistance(distanceFromEnd, totalLength);
  };

  const setTargetPositionsWrapper = useCallback(
    (beads: Bead[]) => {
      return setTargetPositions(beads, BEAD_DIAMETER, getTForDistanceWrapper);
    },
    [getTForDistanceWrapper],
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

    // Adding beads
    if (beadUp > beads.created) {
      setBeads(({ beads, created, destroyed }) => {
        const newBeads = [];
        for (let i = 0; i < beadUp - created; i++) {
          newBeads.push({ t: 0, targetT: 0, startTime: Date.now() });
        }

        const b = setTargetPositionsWrapper([...beads, ...newBeads]);
        return { beads: b, created: beadUp, destroyed };
      });
    }

    const animate = (currentTime: number) => {
      const beads = beadsRef.current;

      if (
        (beadUp === beads.created && beads.created === beads.destroyed) ||
        beads.beads.every((bead) => bead.t >= bead.targetT)
      ) {
        animationFrameRef.current = null;
        return;
      }

      const deltaTime = lastFrameTimeRef.current
        ? currentTime - lastFrameTimeRef.current
        : 16;
      lastFrameTimeRef.current = currentTime;

      setBeads(({ beads, created, destroyed }) => {
        let destroyedCount = 0;

        const newBeads = beads
          .map((bead) => {
            const progressIncrement = deltaTime / ANIMATION_DURATION;
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
          beads: setTargetPositionsWrapper(newBeads),
          created,
          destroyed: destroyed + destroyedCount,
        };
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    lastFrameTimeRef.current = 0;
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [beadUp, beadDown, setTargetPositionsWrapper]);

  return (
    <>
      {beads.beads.map((bead, index) => {
        const pos = getPointAtTWrapper(bead.t);
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

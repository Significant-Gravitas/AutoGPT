// This component uses JS animation [It's replica of legacy builder]
// Bead positions advance every animation frame. They live in a ref and are
// written straight to the circles, so React re-renders only when a bead is
// created or removed, not once per frame.

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  getLengthOfPathInPixels,
  getTForDistance,
  setTargetPositions,
} from "../helpers";

const BEAD_DIAMETER = 10;
const ANIMATION_DURATION = 500;
const FIRST_FRAME_MS = 16;

interface Bead {
  t: number;
  targetT: number;
  startTime: number;
}

interface BeadState {
  beads: Bead[];
  created: number;
  destroyed: number;
}

interface MeasuredPath {
  d: string;
  element: SVGPathElement;
  length: number;
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
  const [beadCount, setBeadCount] = useState(0);
  const stateRef = useRef<BeadState>({ beads: [], created: 0, destroyed: 0 });
  const circlesRef = useRef<(SVGCircleElement | null)[]>([]);
  const edgePathRef = useRef(edgePath);
  const pathRef = useRef<MeasuredPath | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);

  // Measured lazily, once per distinct path: most edges carry no beads.
  function getPath() {
    if (pathRef.current?.d !== edgePathRef.current) {
      pathRef.current = measurePath(edgePathRef.current);
    }
    return pathRef.current;
  }

  function getTForDistanceFromEnd(distanceFromEnd: number) {
    return getTForDistance(distanceFromEnd, getPath().length);
  }

  function positionCircles() {
    const { beads } = stateRef.current;
    if (beads.length === 0) return;
    const path = getPath();
    beads.forEach((bead, index) => {
      const circle = circlesRef.current[index];
      if (!circle) return;
      const point = path.element.getPointAtLength(bead.t * path.length);
      circle.setAttribute("cx", String(point.x));
      circle.setAttribute("cy", String(point.y));
    });
  }

  function commit(next: BeadState) {
    const countChanged = next.beads.length !== stateRef.current.beads.length;
    stateRef.current = next;
    if (countChanged) setBeadCount(next.beads.length);
    positionCircles();
  }

  useLayoutEffect(() => {
    edgePathRef.current = edgePath;
    positionCircles();
  });

  useEffect(() => {
    const current = stateRef.current;
    if (
      beadUp === 0 &&
      beadDown === 0 &&
      (current.created > 0 || current.destroyed > 0)
    ) {
      commit({ beads: [], created: 0, destroyed: 0 });
      return;
    }

    // Adding beads
    if (beadUp > current.created) {
      const newBeads = Array.from({ length: beadUp - current.created }, () => ({
        t: 0,
        targetT: 0,
        startTime: Date.now(),
      }));
      commit({
        beads: setTargetPositions(
          [...current.beads, ...newBeads],
          BEAD_DIAMETER,
          getTForDistanceFromEnd,
        ),
        created: beadUp,
        destroyed: current.destroyed,
      });
    }

    function animate(currentTime: number) {
      const { beads, created, destroyed } = stateRef.current;

      if (
        (beadUp === created && created === destroyed) ||
        beads.every((bead) => bead.t >= bead.targetT)
      ) {
        animationFrameRef.current = null;
        return;
      }

      const deltaTime = lastFrameTimeRef.current
        ? currentTime - lastFrameTimeRef.current
        : FIRST_FRAME_MS;
      lastFrameTimeRef.current = currentTime;

      commit(
        advanceBeads(
          stateRef.current,
          deltaTime,
          beadDown,
          getTForDistanceFromEnd,
        ),
      );

      animationFrameRef.current = requestAnimationFrame(animate);
    }

    lastFrameTimeRef.current = 0;
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [beadUp, beadDown]);

  return (
    <>
      {Array.from({ length: beadCount }, (_, index) => (
        <circle
          key={`${beadsKey}-${index}`}
          ref={(circle) => {
            circlesRef.current[index] = circle;
          }}
          r={BEAD_DIAMETER / 2}
          fill="#8d8d95"
        />
      ))}
    </>
  );
};

function measurePath(d: string): MeasuredPath {
  const element = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "path",
  );
  element.setAttribute("d", d);
  return { d, element, length: getLengthOfPathInPixels(d) };
}

function advanceBeads(
  { beads, created, destroyed }: BeadState,
  deltaTime: number,
  beadDown: number,
  getTForDistanceFromEnd: (distanceFromEnd: number) => number,
): BeadState {
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
    beads: setTargetPositions(newBeads, BEAD_DIAMETER, getTForDistanceFromEnd),
    created,
    destroyed: destroyed + destroyedCount,
  };
}

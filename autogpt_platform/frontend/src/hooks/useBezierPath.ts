import { useCallback, useMemo } from "react";

type XYPosition = {
  x: number;
  y: number;
};

export type BezierPath = {
  sourcePosition: XYPosition;
  control1: XYPosition;
  control2: XYPosition;
  targetPosition: XYPosition;
};

export function useBezierPath(
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
) {
  const path: BezierPath = useMemo(() => {
    const xDifference = Math.abs(sourceX - targetX);
    const yDifference = Math.abs(sourceY - targetY);
    const xControlDistance =
      sourceX < targetX ? 64 : Math.max(xDifference / 2, 64);
    const yControlDistance = yDifference < 128 && sourceX > targetX ? -64 : 0;

    return {
      sourcePosition: { x: sourceX, y: sourceY },
      control1: {
        x: sourceX + xControlDistance,
        y: sourceY + yControlDistance,
      },
      control2: {
        x: targetX - xControlDistance,
        y: targetY + yControlDistance,
      },
      targetPosition: { x: targetX, y: targetY },
    };
  }, [sourceX, sourceY, targetX, targetY]);

  const svgPath = useMemo(
    () =>
      `M ${path.sourcePosition.x} ${path.sourcePosition.y} ` +
      `C ${path.control1.x} ${path.control1.y} ${path.control2.x} ${path.control2.y} ` +
      `${path.targetPosition.x}, ${path.targetPosition.y}`,
    [path],
  );

  const getPointForT = useCallback(
    (t: number) => {
      // Bezier formula: (1-t)^3 * p0 + 3*(1-t)^2*t*p1 + 3*(1-t)*t^2*p2 + t^3*p3
      const x =
        Math.pow(1 - t, 3) * path.sourcePosition.x +
        3 * Math.pow(1 - t, 2) * t * path.control1.x +
        3 * (1 - t) * Math.pow(t, 2) * path.control2.x +
        Math.pow(t, 3) * path.targetPosition.x;

      const y =
        Math.pow(1 - t, 3) * path.sourcePosition.y +
        3 * Math.pow(1 - t, 2) * t * path.control1.y +
        3 * (1 - t) * Math.pow(t, 2) * path.control2.y +
        Math.pow(t, 3) * path.targetPosition.y;

      return { x, y };
    },
    [path],
  );

  const getArcLength = useCallback(
    (t: number, samples: number = 100) => {
      let length = 0;
      let prevPoint = getPointForT(0);

      for (let i = 1; i <= samples; i++) {
        const currT = (i / samples) * t;
        const currPoint = getPointForT(currT);
        length += Math.sqrt(
          Math.pow(currPoint.x - prevPoint.x, 2) +
            Math.pow(currPoint.y - prevPoint.y, 2),
        );
        prevPoint = currPoint;
      }

      return length;
    },
    [getPointForT],
  );

  const length = useMemo(() => {
    return getArcLength(1);
  }, [getArcLength]);

  const getBezierDerivative = useCallback(
    (t: number) => {
      const mt = 1 - t;
      const x =
        3 *
        (mt * mt * (path.control1.x - path.sourcePosition.x) +
          2 * mt * t * (path.control2.x - path.control1.x) +
          t * t * (path.targetPosition.x - path.control2.x));
      const y =
        3 *
        (mt * mt * (path.control1.y - path.sourcePosition.y) +
          2 * mt * t * (path.control2.y - path.control1.y) +
          t * t * (path.targetPosition.y - path.control2.y));
      return { x, y };
    },
    [path],
  );

  const getTForDistance = useCallback(
    (distance: number, epsilon: number = 0.0001) => {
      if (distance < 0) {
        distance = length + distance; // If distance is negative, calculate from the end of the curve
      }

      let t = distance / getArcLength(1);
      let prevT = 0;

      while (Math.abs(t - prevT) > epsilon) {
        prevT = t;
        const length = getArcLength(t);
        const derivative = Math.sqrt(
          Math.pow(getBezierDerivative(t).x, 2) +
            Math.pow(getBezierDerivative(t).y, 2),
        );
        t -= (length - distance) / derivative;
        t = Math.max(0, Math.min(1, t)); // Clamp t between 0 and 1
      }

      return t;
    },
    [getArcLength, getBezierDerivative, length],
  );

  const getPointAtDistance = useCallback(
    (distance: number) => {
      if (distance < 0) {
        distance = length + distance; // If distance is negative, calculate from the end of the curve
      }

      const t = getTForDistance(distance);
      return getPointForT(t);
    },
    [getTForDistance, getPointForT, length],
  );

  return {
    path,
    svgPath,
    length,
    getPointForT,
    getTForDistance,
    getPointAtDistance,
  };
}

import React, { FC, memo, useEffect, useMemo, useState } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  EdgeProps,
  useReactFlow,
  XYPosition,
} from "reactflow";
import "./customedge.css";
import { X } from "lucide-react";

export type CustomEdgeData = {
  edgeColor: string;
  sourcePos?: XYPosition;
  beadUp?: number;
  beadDown?: number;
  beadData?: Array<any>;
};

type BezierPath = {
  sourcePosition: XYPosition;
  control1: XYPosition;
  control2: XYPosition;
  targetPosition: XYPosition;
}

type Bead = {
  t: number;
  targetT: number;
  startTime: number;
}

const CustomEdgeFC: FC<EdgeProps<CustomEdgeData>> = ({
  id,
  data,
  selected,
  source,
  sourcePosition,
  sourceX,
  sourceY,
  target,
  targetPosition,
  targetX,
  targetY,
  markerEnd,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const { setEdges } = useReactFlow();
  const [beads, setBeads] = useState<{ beads: Array<Bead>, created: number, destroyed: number }>({ beads: [], created: 0, destroyed: 0 });

  // Calculate y difference between source and source node, to adjust self-loop edge
  const yDifference = useMemo(
    () => sourceY - (data?.sourcePos?.y || 0),
    [data?.sourcePos?.y],
  );

  const path = source === target ?
    {
      sourcePosition: { x: sourceX - 5, y: sourceY },
      control1: { x: sourceX + 128, y: sourceY - yDifference - 128 },
      control2: { x: targetX - 128, y: sourceY - yDifference - 128 },
      targetPosition: { x: targetX + 3, y: targetY },
    } :
    {
      sourcePosition: { x: sourceX - 5, y: sourceY },
      control1: { x: sourceX + 128, y: sourceY },
      control2: { x: targetX - 128, y: targetY },
      targetPosition: { x: targetX + 3, y: targetY },
    };

  const onEdgeRemoveClick = () => {
    setEdges((edges) => edges.filter((edge) => edge.id !== id));
  };

  const ANIMATION_DURATION = 2000; // Duration in milliseconds for bead to travel the curve
  const beadDiameter = 10;

  function setTargetPositions(path: BezierPath, beads: Bead[]) {
    const pathlength = getBezierLength(path);
    const distanceBetween = Math.min((pathlength - beadDiameter - 5) / (beads.length + 1), beadDiameter + 5);

    return beads.map((bead, index) => {
      const targetPosition = distanceBetween * (index + 1);
      const t = getTForDistance(path, -targetPosition);

      return {
        ...bead,
        targetT: t,
      } as Bead;
    });
  }

  useEffect(() => {
    if (data?.beadUp === 0 && data?.beadDown === 0) {
      setBeads({ beads: [], created: 0, destroyed: 0 });
      return
    }

    //FIXME: This is a temporary fix, bead count is twice as many
    const beadUp = data?.beadUp! / 2;

    setBeads(({ beads, created, destroyed }) => {
      const newBeads = [];
      for (let i = 0; i < beadUp - created; i++) {
        newBeads.push({ t: 0, targetT: 0, startTime: Date.now() });
      }

      const b = setTargetPositions(path, [...beads, ...newBeads])
      return { beads: b, created: beadUp, destroyed };
    });

    const interval = setInterval(() => {

      setBeads(({ beads, created, destroyed }) => {

        let destroyedCount = 0;

        const newBeads = beads.map((bead) => {
          const elapsed = Date.now() - bead.startTime;
          const progress = Math.min(elapsed / ANIMATION_DURATION, 1);
          const t = bead.t + (bead.targetT - bead.t) * progress;

          return {
            ...bead,
            t,
          };

        }).filter((bead, index, beads) => {
          const beadDown = data?.beadDown! / 2;
          const length = beads.length

          const removeCount = beadDown - destroyed
          if (bead.t >= bead.targetT && index >= length - removeCount) {
            destroyedCount++;
            return false;
          }
          return true;
        })

        return { beads: newBeads, created, destroyed: destroyed + destroyedCount };
      })

    }, 16); // 60fps animation

    return () => clearInterval(interval);
  }, [data]);

  function getBezierPathString(path: BezierPath) {
    return `M ${path.sourcePosition.x} ${path.sourcePosition.y} ` +
      `C ${path.control1.x} ${path.control1.y} ${path.control2.x} ${path.control2.y} ` +
      `${path.targetPosition.x}, ${path.targetPosition.y}`;
  }

  //todo kcze memo
  function getBezierLength(path: BezierPath, steps: number = 100): number {
    let length = 0;
    let previousPoint = getBezierPoint(0, path);

    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      const currentPoint = getBezierPoint(t, path);
      length += Math.sqrt(
        Math.pow(currentPoint.x - previousPoint.x, 2) + Math.pow(currentPoint.y - previousPoint.y, 2)
      );
      previousPoint = currentPoint;
    }

    return length;
  }

  function getTForDistance(path: BezierPath, distance: number, steps: number = 100): number {
    const length = getBezierLength(path, steps);

    if (distance < 0) {
      distance = length + distance; // If distance is negative, calculate from the end of the curve
    }

    return distance / length; // If the distance exceeds the curve length, return the endpoint
  }

  function getBezierPoint(t: number, path: BezierPath) {
    // Bezier formula: (1-t)^3 * p0 + 3*(1-t)^2*t*p1 + 3*(1-t)*t^2*p2 + t^3*p3
    const x = Math.pow(1 - t, 3) * path.sourcePosition.x +
      3 * Math.pow(1 - t, 2) * t * path.control1.x +
      3 * (1 - t) * Math.pow(t, 2) * path.control2.x +
      Math.pow(t, 3) * path.targetPosition.x;

    const y = Math.pow(1 - t, 3) * path.sourcePosition.y +
      3 * Math.pow(1 - t, 2) * t * path.control1.y +
      3 * (1 - t) * Math.pow(t, 2) * path.control2.y +
      Math.pow(t, 3) * path.targetPosition.y;

    return { x, y };
  }

  function getBezierPointAtDistance(distance: number, path: BezierPath) {
    const t = getTForDistance(path, distance);
    return getBezierPoint(t, path);
  }

  const middle = getBezierPoint(0.5, path);

  return (
    <>
      <BaseEdge
        path={getBezierPathString(path)}
        markerEnd={markerEnd}
        style={{
          strokeWidth: isHovered ? 3 : 2,
          stroke:
            (data?.edgeColor ?? "#555555") +
            (selected || isHovered ? "" : "80"),
        }}
      />
      <path
        d={getBezierPathString(path)}
        fill="none"
        strokeOpacity={0}
        strokeWidth={20}
        className="react-flow__edge-interaction"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      />
      <EdgeLabelRenderer>
        <div
          style={{
            position: "absolute",
            transform: `translate(-50%, -50%) translate(${middle.x}px,${middle.y}px)`,
            pointerEvents: "all",
          }}
          className="edge-label-renderer"
        >
          <button
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            className={`edge-label-button ${isHovered ? "visible" : ""}`}
            onClick={onEdgeRemoveClick}
          >
            <X className="size-4" />
          </button>
          <span>{beads.beads.length}⚫️ {data?.beadUp! / 2}🔼 {data?.beadDown! / 2}🔽</span>
        </div>
      </EdgeLabelRenderer>
      {beads.beads.map((bead) => {
        const pos = getBezierPoint(bead.t, path);
        return (<circle
          cx={pos.x}
          cy={pos.y}
          r={beadDiameter / 2} // Bead radius
          fill={data?.edgeColor ?? "#555555"}
        />)
      })}
    </>
  );
};

export const CustomEdge = memo(CustomEdgeFC);

import React, { FC, memo, useEffect, useMemo, useState } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  EdgeProps,
  Position,
  useReactFlow,
  XYPosition,
} from "reactflow";
import "./customedge.css";
import { X } from "lucide-react";
import { useBezierPath } from "@/hooks/useBezierPath";

export type CustomEdgeData = {
  edgeColor: string;
  sourcePos?: XYPosition;
  beadUp?: number;
  beadDown?: number;
  beadData?: any[];
};

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
  const [beads, setBeads] = useState<{ beads: Bead[], created: number, destroyed: number }>({ beads: [], created: 0, destroyed: 0 });
  const { path, svgPath, length, getPointForT, getTForDistance, getPointAtDistance } = 
    useBezierPath(sourceX - 5, sourceY, targetX + 3, targetY);

  const onEdgeRemoveClick = () => {
    setEdges((edges) => edges.filter((edge) => edge.id !== id));
  };

  const animation_duration = 500; // Duration in milliseconds for bead to travel the curve
  const beadDiameter = 12;
  const delta_time = 16;

  function setTargetPositions(beads: Bead[]) {
    const distanceBetween = Math.min((length - beadDiameter) / (beads.length + 1), beadDiameter);

    return beads.map((bead, index) => {
      const targetPosition = distanceBetween * (index + 1);
      const t = getTForDistance(-targetPosition);

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

      const b = setTargetPositions([...beads, ...newBeads])
      return { beads: b, created: beadUp, destroyed };
    });

    const interval = setInterval(() => {

      setBeads(({ beads, created, destroyed }) => {

        let destroyedCount = 0;

        const newBeads = beads.map((bead) => {
          // const elapsed = Date.now() - bead.startTime;
          // const progress = Math.min(elapsed / ANIMATION_DURATION, 1);
          // const t = bead.t + (bead.targetT - bead.t) * progress;
          const progressIncrement = delta_time / animation_duration;
          const t = Math.min(bead.t + bead.targetT * progressIncrement, bead.targetT);

          return {
            ...bead,
            t,
          };

        }).filter((bead, index, beads) => {
          const beadDown = data?.beadDown! / 2;

          const removeCount = beadDown - destroyed
          if (bead.t >= bead.targetT && index < removeCount) {
            destroyedCount++;
            return false;
          }
          return true;
        })

        return { beads: setTargetPositions(newBeads), created, destroyed: destroyed + destroyedCount };
      })

    }, delta_time); // 60fps animation

    return () => clearInterval(interval);
  }, [data]);

  const middle = getPointForT(0.5);

  return (
    <>
      <BaseEdge
        path={svgPath}
        markerEnd={markerEnd}
        style={{
          strokeWidth: isHovered ? 3 : 2,
          stroke:
            (data?.edgeColor ?? "#555555") +
            (selected || isHovered ? "" : "80"),
        }}
      />
      <path
        d={svgPath}
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
      {beads.beads.map((bead, index) => {
        const pos = getPointForT(bead.t);
        return (<circle
          key={index}
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

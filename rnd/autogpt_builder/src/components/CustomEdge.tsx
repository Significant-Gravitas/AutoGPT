import React, { FC, memo, useContext, useEffect, useState } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  EdgeProps,
  useReactFlow,
  XYPosition,
} from "reactflow";
import "./customedge.css";
import { X } from "lucide-react";
import { useBezierPath } from "@/hooks/useBezierPath";
import { FlowContext } from "./Flow";

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
};

const CustomEdgeFC: FC<EdgeProps<CustomEdgeData>> = ({
  id,
  data,
  selected,
  sourceX,
  sourceY,
  targetX,
  targetY,
  markerEnd,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [beads, setBeads] = useState<{
    beads: Bead[];
    created: number;
    destroyed: number;
  }>({ beads: [], created: 0, destroyed: 0 });
  const { svgPath, length, getPointForT, getTForDistance } = useBezierPath(
    sourceX - 5,
    sourceY,
    targetX + 3,
    targetY,
  );
  const { deleteElements } = useReactFlow<any, CustomEdgeData>();
  const { visualizeBeads } = useContext(FlowContext) ?? {
    visualizeBeads: "no",
  };

  const onEdgeRemoveClick = () => {
    deleteElements({ edges: [{ id }] });
  };

  const animationDuration = 500; // Duration in milliseconds for bead to travel the curve
  const beadDiameter = 10;
  const deltaTime = 16;

  function setTargetPositions(beads: Bead[]) {
    const distanceBetween = Math.min(
      (length - beadDiameter) / (beads.length + 1),
      beadDiameter,
    );

    return beads.map((bead, index) => {
      const targetPosition = distanceBetween * index + beadDiameter * 1.3;
      const t = getTForDistance(-targetPosition);

      return {
        ...bead,
        t: visualizeBeads === "animate" ? bead.t : t,
        targetT: t,
      } as Bead;
    });
  }

  useEffect(() => {
    if (data?.beadUp === 0 && data?.beadDown === 0) {
      setBeads({ beads: [], created: 0, destroyed: 0 });
      return;
    }

    const beadUp = data?.beadUp!;

    setBeads(({ beads, created, destroyed }) => {
      const newBeads = [];
      for (let i = 0; i < beadUp - created; i++) {
        newBeads.push({ t: 0, targetT: 0, startTime: Date.now() });
      }

      const b = setTargetPositions([...beads, ...newBeads]);
      return { beads: b, created: beadUp, destroyed };
    });

    if (visualizeBeads !== "animate") {
      setBeads(({ beads, created, destroyed }) => {
        let destroyedCount = 0;

        const newBeads = beads
          .map((bead) => ({ ...bead }))
          .filter((bead, index) => {
            const beadDown = data?.beadDown!;

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
      return;
    }

    const interval = setInterval(() => {
      setBeads(({ beads, created, destroyed }) => {
        let destroyedCount = 0;

        const newBeads = beads
          .map((bead) => {
            const progressIncrement = deltaTime / animationDuration;
            const t = Math.min(
              bead.t + bead.targetT * progressIncrement,
              bead.targetT,
            );

            return {
              ...bead,
              t,
            };
          })
          .filter((bead, index) => {
            const beadDown = data?.beadDown!;

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
    }, deltaTime);

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
        </div>
      </EdgeLabelRenderer>
      {beads.beads.map((bead, index) => {
        const pos = getPointForT(bead.t);
        return (
          <circle
            key={index}
            cx={pos.x}
            cy={pos.y}
            r={beadDiameter / 2} // Bead radius
            fill={data?.edgeColor ?? "#555555"}
          />
        );
      })}
    </>
  );
};

export const CustomEdge = memo(CustomEdgeFC);

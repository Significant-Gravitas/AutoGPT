import React, {
  useCallback,
  useContext,
  useEffect,
  useState,
  useRef,
} from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  EdgeProps,
  useReactFlow,
  XYPosition,
  Edge,
  Node,
} from "@xyflow/react";
import "./customedge.css";
import { X } from "lucide-react";
import { BuilderContext } from "../Flow/Flow";
import { NodeExecutionResult } from "@/lib/autogpt-server-api";
import { useCustomEdge } from "./useCustomEdge";

export type CustomEdgeData = {
  edgeColor: string;
  sourcePos?: XYPosition;
  isStatic?: boolean;
  beadUp: number;
  beadDown: number;
  beadData?: Map<string, NodeExecutionResult["status"]>;
};

type Bead = {
  t: number;
  targetT: number;
  startTime: number;
};

export type CustomEdge = Edge<CustomEdgeData, "custom">;

export function CustomEdge({
  id,
  data,
  selected,
  sourceX,
  sourceY,
  targetX,
  targetY,
  markerEnd,
}: EdgeProps<CustomEdge>) {
  const [beads, setBeads] = useState<{
    beads: Bead[];
    created: number;
    destroyed: number;
  }>({ beads: [], created: 0, destroyed: 0 });
  const beadsRef = useRef(beads);
  const { svgPath, length, getPointForT, getTForDistance } = useCustomEdge(
    sourceX - 5,
    sourceY - 5,
    targetX + 3,
    targetY - 5,
  );
  const { deleteElements } = useReactFlow<Node, CustomEdge>();
  const { visualizeBeads } = useContext(BuilderContext) ?? {
    visualizeBeads: "no",
  };

  const onEdgeRemoveClick = () => {
    deleteElements({ edges: [{ id }] });
  };

  const animationDuration = 500; // Duration in milliseconds for bead to travel the curve
  const beadDiameter = 12;
  const deltaTime = 16;

  const setTargetPositions = useCallback(
    (beads: Bead[]) => {
      const distanceBetween = Math.min(
        (length - beadDiameter) / (beads.length + 1),
        beadDiameter,
      );

      return beads.map((bead, index) => {
        const distanceFromEnd = beadDiameter * 1.35;
        const targetPosition = distanceBetween * index + distanceFromEnd;
        const t = getTForDistance(-targetPosition);

        return {
          ...bead,
          t: visualizeBeads === "animate" ? bead.t : t,
          targetT: t,
        } as Bead;
      });
    },
    [getTForDistance, length, visualizeBeads],
  );

  beadsRef.current = beads;
  useEffect(() => {
    const beadUp: number = data?.beadUp ?? 0;
    const beadDown: number = data?.beadDown ?? 0;

    if (
      beadUp === 0 &&
      beadDown === 0 &&
      (beads.created > 0 || beads.destroyed > 0)
    ) {
      setBeads({ beads: [], created: 0, destroyed: 0 });
      return;
    }

    // Add beads
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

    // Animate and remove beads
    const interval = setInterval(
      ({ current: beads }) => {
        // If there are no beads visible or moving, stop re-rendering
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
              const progressIncrement = deltaTime / animationDuration;
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
      deltaTime,
      beadsRef,
    );

    return () => clearInterval(interval);
  }, [data?.beadUp, data?.beadDown, setTargetPositions, visualizeBeads]);

  const middle = getPointForT(0.5);

  return (
    <>
      <BaseEdge
        path={svgPath}
        markerEnd={markerEnd}
        className={`data-sentry-unmask transition-all duration-200 ${data?.isStatic ? "[stroke-dasharray:5_3]" : "[stroke-dasharray:0]"} [stroke-width:${data?.isStatic ? 2.5 : 2}px] hover:[stroke-width:${data?.isStatic ? 3.5 : 3}px] ${selected ? `[stroke:${data?.edgeColor ?? "#555555"}]` : `[stroke:${data?.edgeColor ?? "#555555"}80] hover:[stroke:${data?.edgeColor ?? "#555555"}]`}`}
      />
      <path
        d={svgPath}
        fill="none"
        strokeOpacity={0}
        strokeWidth={20}
        className="data-sentry-unmask react-flow__edge-interaction"
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
            className="edge-label-button opacity-0 transition-opacity duration-200 hover:opacity-100"
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
}

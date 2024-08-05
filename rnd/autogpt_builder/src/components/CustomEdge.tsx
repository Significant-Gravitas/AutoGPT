import React, { FC, memo, useMemo, useState } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  EdgeProps,
  getBezierPath,
  useReactFlow,
  XYPosition,
} from "reactflow";
import "./customedge.css";
import { X } from "lucide-react";

export type CustomEdgeData = {
  edgeColor: string;
  sourcePos?: XYPosition;
};

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

  const onEdgeClick = () => {
    setEdges((edges) => edges.filter((edge) => edge.id !== id));
    data.clearNodesStatusAndOutput();
  };

  const [path, labelX, labelY] = getBezierPath({
    sourceX: sourceX - 5,
    sourceY,
    sourcePosition,
    targetX: targetX + 4,
    targetY,
    targetPosition,
  });

  // Calculate y difference between source and source node, to adjust self-loop edge
  const yDifference = useMemo(
    () => sourceY - (data?.sourcePos?.y || 0),
    [data?.sourcePos?.y],
  );

  // Define special edge path for self-loop
  const edgePath =
    source === target
      ? `M ${sourceX - 5} ${sourceY} C ${sourceX + 128} ${sourceY - yDifference - 128} ${targetX - 128} ${sourceY - yDifference - 128} ${targetX + 3}, ${targetY}`
      : path;

  console.table({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    path,
    labelX,
    labelY,
  });

  return (
    <>
      <BaseEdge
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          strokeWidth: isHovered ? 3 : 2,
          stroke:
            (data?.edgeColor ?? "#555555") +
            (selected || isHovered ? "" : "80"),
        }}
      />
      <path
        d={edgePath}
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
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            pointerEvents: "all",
          }}
          className="edge-label-renderer"
        >
          <button
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            className={`edge-label-button ${isHovered ? "visible" : ""}`}
            onClick={onEdgeClick}
          >
            <X className="size-4" />
          </button>
        </div>
      </EdgeLabelRenderer>
    </>
  );
};

export const CustomEdge = memo(CustomEdgeFC);

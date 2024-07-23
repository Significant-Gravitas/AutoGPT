import { FC, memo, useMemo } from "react";
import { BaseEdge, EdgeProps, getBezierPath, XYPosition } from "reactflow";

export type CustomEdgeData = {
  edgeColor: string
  sourcePos: XYPosition
}

const CustomEdgeFC: FC<EdgeProps<CustomEdgeData>> = ({ data, selected, source, sourcePosition, sourceX, sourceY, target, targetPosition, targetX, targetY, markerEnd }) => {

  const [path] = getBezierPath({
    sourceX: sourceX - 5,
    sourceY,
    sourcePosition,
    targetX: targetX + 4,
    targetY,
    targetPosition,
  });

  // Calculate y difference between source and source node, to adjust self-loop edge
  const yDifference = useMemo(() => sourceY - data!.sourcePos.y, [data!.sourcePos.y]);

  // Define special edge path for self-loop
  const edgePath = source === target ?
    `M ${sourceX - 5} ${sourceY} C ${sourceX + 128} ${sourceY - yDifference - 128} ${targetX - 128} ${sourceY - yDifference - 128} ${targetX + 3}, ${targetY}` :
    path;

  return (
    <BaseEdge
      style={{ strokeWidth: 2, stroke: (data?.edgeColor ?? '#555555') + (selected ? '' : '80') }}
      path={edgePath}
      markerEnd={markerEnd}
    />
  )
};

export const CustomEdge = memo(CustomEdgeFC);

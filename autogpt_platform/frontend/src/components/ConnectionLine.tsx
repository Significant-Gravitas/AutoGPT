import {
  BaseEdge,
  ConnectionLineComponentProps,
  Node,
  getBezierPath,
  Position,
} from "@xyflow/react";

export default function ConnectionLine<NodeType extends Node>({
  fromPosition,
  fromHandle,
  fromX,
  fromY,
  toPosition,
  toX,
  toY,
}: ConnectionLineComponentProps<NodeType>) {
  const sourceX =
    fromPosition === Position.Right
      ? fromX + (fromHandle?.width! / 2 - 5)
      : fromX - (fromHandle?.width! / 2 - 5);

  const [path] = getBezierPath({
    sourceX: sourceX,
    sourceY: fromY,
    sourcePosition: fromPosition,
    targetX: toX,
    targetY: toY,
    targetPosition: toPosition,
  });

  return <BaseEdge path={path} style={{ strokeWidth: 2, stroke: "#555" }} />;
}

import {
  BaseEdge,
  ConnectionLineComponentProps,
  getBezierPath,
  Position,
} from "@xyflow/react";

const ConnectionLine: React.FC<ConnectionLineComponentProps> = ({
  fromPosition,
  fromHandle,
  fromX,
  fromY,
  toPosition,
  toX,
  toY,
}) => {
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
};

export default ConnectionLine;

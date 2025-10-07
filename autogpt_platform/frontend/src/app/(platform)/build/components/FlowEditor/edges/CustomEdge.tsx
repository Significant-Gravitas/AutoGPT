import { Button } from "@/components/atoms/Button/Button";
import {
  BaseEdge,
  EdgeLabelRenderer,
  EdgeProps,
  getBezierPath,
} from "@xyflow/react";

import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { XIcon } from "@phosphor-icons/react";

const CustomEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  selected,
}: EdgeProps) => {
  const removeConnection = useEdgeStore((state) => state.removeConnection);
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  return (
    <>
      <BaseEdge
        path={edgePath}
        markerEnd={markerEnd}
        className={
          selected ? "[stroke:#555]" : "[stroke:#555]80 hover:[stroke:#555]"
        }
      />
      <EdgeLabelRenderer>
        <Button
          onClick={() => removeConnection(id)}
          className={`absolute z-10 h-fit min-w-0 p-1`}
          variant="secondary"
          style={{
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            pointerEvents: "all",
          }}
        >
          <XIcon className="h-3 w-3" weight="bold" />
        </Button>
      </EdgeLabelRenderer>
    </>
  );
};

export default CustomEdge;

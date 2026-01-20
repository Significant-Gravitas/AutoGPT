import { memo, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  BaseEdge,
  Edge as XYEdge,
  EdgeLabelRenderer,
  EdgeProps,
  getBezierPath,
} from "@xyflow/react";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { XIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { NodeExecutionResult } from "@/lib/autogpt-server-api";
import { JSBeads } from "./components/JSBeads";

export type CustomEdgeData = {
  isStatic?: boolean;
  beadUp?: number;
  beadDown?: number;
  beadData?: Map<string, NodeExecutionResult["status"]>;
  edgeColorClass?: string;
  edgeHexColor?: string;
};

export type CustomEdge = XYEdge<CustomEdgeData, "custom">;

const CustomEdge = ({
  id,
  data,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  selected,
}: EdgeProps<CustomEdge>) => {
  const removeConnection = useEdgeStore((state) => state.removeEdge);
  const isBroken = useNodeStore((state) => state.isEdgeBroken(id));
  const [isHovered, setIsHovered] = useState(false);

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const isStatic = data?.isStatic ?? false;
  const beadUp = data?.beadUp ?? 0;
  const beadDown = data?.beadDown ?? 0;
  const edgeColorClass = data?.edgeColorClass;

  const handleRemoveEdge = () => {
    removeConnection(id);
    // Note: broken edge tracking is cleaned up automatically by useSubAgentUpdateState
    // when it detects the edge no longer exists
  };

  return (
    <>
      <BaseEdge
        path={edgePath}
        markerEnd={markerEnd}
        className={cn(
          isStatic && "!stroke-[1.5px] [stroke-dasharray:6]",
          isBroken
            ? "!stroke-red-500 !stroke-[2px] [stroke-dasharray:4]"
            : selected
              ? "stroke-zinc-800"
              : edgeColorClass
                ? cn(edgeColorClass, "opacity-70 hover:opacity-100")
                : "stroke-zinc-500/50 hover:stroke-zinc-500",
        )}
      />
      <JSBeads
        beadUp={beadUp}
        beadDown={beadDown}
        edgePath={edgePath}
        beadsKey={`beads-${id}-${sourceX}-${sourceY}-${targetX}-${targetY}`}
      />
      <EdgeLabelRenderer>
        <Button
          onClick={handleRemoveEdge}
          className={cn(
            "absolute h-fit min-w-0 p-1 transition-opacity",
            isBroken
              ? "bg-red-500 opacity-100 hover:bg-red-600"
              : isHovered
                ? "opacity-100"
                : "opacity-0",
          )}
          variant={isBroken ? "primary" : "secondary"}
          style={{
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            pointerEvents: "all",
          }}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <XIcon className="h-3 w-3" weight="bold" />
        </Button>
      </EdgeLabelRenderer>
    </>
  );
};

export default memo(CustomEdge);

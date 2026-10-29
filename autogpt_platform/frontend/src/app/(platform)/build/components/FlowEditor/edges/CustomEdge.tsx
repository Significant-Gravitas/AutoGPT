import { memo, useState, type CSSProperties } from "react";
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
import { cn } from "@/lib/utils";
import { NodeExecutionResult } from "@/lib/autogpt-server-api";
import { JSBeads } from "./components/JSBeads";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
      <path
        d={edgePath}
        fill="none"
        stroke="currentColor"
        strokeOpacity={0}
        strokeWidth={20}
        // oxlint-disable-next-line design-system/no-unknown-classes -- xyflow's own class for an edge's hit area
        className="react-flow__edge-interaction cursor-pointer"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      />
      <BaseEdge
        path={edgePath}
        interactionWidth={0}
        markerEnd={markerEnd}
        className={cn(
          // oxlint-disable-next-line design-system/no-arbitrary-values -- a static edge is 1.5px; the stroke-width scale only has 0, 1 and 2
          isStatic && "!stroke-[1.5px] [stroke-dasharray:6]",
          isBroken
            ? "!stroke-error !stroke-2 [stroke-dasharray:4]"
            : selected
              ? "stroke-foreground-emphasis"
              : edgeColorClass
                ? cn(edgeColorClass, "opacity-70 hover:opacity-100")
                : "stroke-edge/50 hover:stroke-edge",
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
            "pointer-events-auto absolute left-[var(--label-x)] top-[var(--label-y)] h-fit min-w-0 -translate-x-1/2 -translate-y-1/2 p-1 transition-opacity",
            isBroken
              ? "bg-error opacity-100 hover:bg-error-hover"
              : isHovered
                ? "opacity-100"
                : "opacity-0",
          )}
          variant={isBroken ? "primary" : "secondary"}
          style={
            {
              "--label-x": `${labelX}px`,
              "--label-y": `${labelY}px`,
            } as CSSProperties
          }
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <Icon icon={Cancel01Icon} className="h-3 w-3" />
        </Button>
      </EdgeLabelRenderer>
    </>
  );
};

export default memo(CustomEdge);

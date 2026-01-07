import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CaretDownIcon, InfoIcon } from "@phosphor-icons/react";
import { RJSFSchema } from "@rjsf/utils";
import { useMemo, useState } from "react";

import NodeHandle from "../handlers/NodeHandle";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { getTypeDisplayInfo } from "./helpers";
import { generateHandleId } from "../handlers/helpers";
import { BlockUIType } from "../../types";
import { cn } from "@/lib/utils";

/**
 * Hook to get the set of broken output names for a node in resolution mode.
 */
function useBrokenOutputs(nodeID: string): Set<string> {
  // Subscribe to the actual state values, not just methods
  const isInResolution = useNodeStore((state) =>
    state.nodesInResolutionMode.has(nodeID),
  );
  const resolutionData = useNodeStore((state) =>
    state.nodeResolutionData.get(nodeID),
  );

  return useMemo(() => {
    if (!isInResolution || !resolutionData) {
      return new Set<string>();
    }

    return new Set(resolutionData.incompatibilities.missingOutputs);
  }, [isInResolution, resolutionData]);
}

export const OutputHandler = ({
  outputSchema,
  nodeId,
  uiType,
}: {
  outputSchema: RJSFSchema;
  nodeId: string;
  uiType: BlockUIType;
}) => {
  const { isOutputConnected } = useEdgeStore();
  const properties = outputSchema?.properties || {};
  const [isOutputVisible, setIsOutputVisible] = useState(true);
  const brokenOutputs = useBrokenOutputs(nodeId);

  return (
    <div className="flex flex-col items-end justify-between gap-2 rounded-b-xlarge border-t border-slate-200/50 bg-white py-3.5">
      <Button
        variant="ghost"
        className="mr-4 h-fit min-w-0 p-0 hover:border-transparent hover:bg-transparent"
        onClick={() => setIsOutputVisible(!isOutputVisible)}
      >
        <Text
          variant="body"
          className="flex items-center gap-2 !font-semibold text-slate-700"
        >
          Output{" "}
          <CaretDownIcon
            size={16}
            weight="bold"
            className={`transition-transform ${isOutputVisible ? "rotate-180" : ""}`}
          />
        </Text>
      </Button>

      {
        <div className="flex flex-col items-end gap-2">
          {Object.entries(properties).map(([key, property]: [string, any]) => {
            const isConnected = isOutputConnected(nodeId, key);
            const isBroken = brokenOutputs.has(key);
            const shouldShow = isConnected || isOutputVisible;
            const { displayType, colorClass } = getTypeDisplayInfo(property);

            return shouldShow ? (
              <div key={key} className="relative flex items-center gap-2">
                {property?.description && (
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span
                          style={{ marginLeft: 6, cursor: "pointer" }}
                          aria-label="info"
                          tabIndex={0}
                        >
                          <InfoIcon />
                        </span>
                      </TooltipTrigger>
                      <TooltipContent>{property?.description}</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}
                <Text
                  variant="body"
                  className={cn(
                    "text-slate-700",
                    isBroken && "text-red-500 line-through",
                  )}
                >
                  {property?.title || key}{" "}
                </Text>
                <Text variant="small" as="span" className={colorClass}>
                  ({displayType})
                </Text>

                <NodeHandle
                  handleId={
                    uiType === BlockUIType.AGENT ? key : generateHandleId(key)
                  }
                  isConnected={isConnected}
                  side="right"
                  isBroken={isBroken}
                />
              </div>
            ) : null;
          })}
        </div>
      }
    </div>
  );
};

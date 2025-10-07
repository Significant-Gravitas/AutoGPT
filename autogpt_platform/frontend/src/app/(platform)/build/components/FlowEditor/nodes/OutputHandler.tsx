import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CaretDownIcon, InfoIcon } from "@phosphor-icons/react";
import { RJSFSchema } from "@rjsf/utils";
import { useState } from "react";

import NodeHandle from "../handlers/NodeHandle";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { getTypeDisplayInfo } from "./helpers";

export const OutputHandler = ({
  outputSchema,
  nodeId,
}: {
  outputSchema: RJSFSchema;
  nodeId: string;
}) => {
  const { isOutputConnected } = useEdgeStore();
  const properties = outputSchema?.properties || {};
  const [isOutputVisible, setIsOutputVisible] = useState(false);

  return (
    <div className="flex flex-col items-end justify-between gap-2 rounded-b-xl border-t border-slate-200/50 bg-white py-3.5">
      <Button
        variant="ghost"
        className="mr-4 h-fit min-w-0 p-0 hover:border-transparent hover:bg-transparent"
        onClick={() => setIsOutputVisible(!isOutputVisible)}
      >
        <Text
          variant="body"
          className="flex items-center gap-2 font-medium text-slate-700"
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
                <Text variant="body" className="text-slate-700">
                  {property?.title || key}{" "}
                </Text>
                <Text variant="small" as="span" className={colorClass}>
                  ({displayType})
                </Text>
                <NodeHandle id={key} isConnected={isConnected} side="right" />
              </div>
            ) : null;
          })}
        </div>
      }
    </div>
  );
};

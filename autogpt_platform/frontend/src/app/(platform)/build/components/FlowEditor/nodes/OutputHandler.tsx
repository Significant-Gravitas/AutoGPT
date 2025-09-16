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
} from "@/components/ui/tooltip";
import { useEdgeStore } from "../../store/edgeStore";

export const OutputHandler = ({
  outputSchema,
  nodeId,
}: {
  outputSchema: RJSFSchema;
  nodeId: string;
}) => {
  const { isOutputConnected } = useEdgeStore();

  // Extract properties from the schema
  const properties = outputSchema?.properties || {};

  // State to control visibility of output properties
  const [isOutputVisible, setIsOutputVisible] = useState(false);

  return (
    <div className="flex flex-col items-end justify-between gap-2 rounded-b-xl border-t border-slate-200/50 bg-white py-3.5">
      <Button
        variant="ghost"
        className="mr-4 p-0"
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

      {/* Output Properties */}
      {isOutputVisible && (
        <div className="flex flex-col items-end gap-2">
          {Object.entries(properties).map(([key, property]: [string, any]) => (
            <div key={key} className="relative flex items-center gap-2">
              <Text
                variant="body"
                className="flex items-center gap-2 font-medium text-slate-700"
              >
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
                {property?.title || key}{" "}
                <Text variant="small" as="span" className="!text-green-500">
                  ({property?.type || "unknown"})
                </Text>
              </Text>
              <NodeHandle
                id={key}
                isConnected={isOutputConnected(nodeId, key)}
                side="right"
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

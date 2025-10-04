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
  const [expandedObjects, setExpandedObjects] = useState<Set<string>>(new Set());

  // Helper function to get the parent object key for a sub-output
  const getParentKey = (key: string) => {
    if (key.includes('.')) {
      return key.split('.')[0];
    }
    return null;
  };

  // Helper function to toggle object expansion
  const toggleObjectExpansion = (key: string) => {
    setExpandedObjects(prev => {
      const newSet = new Set(prev);
      if (newSet.has(key)) {
        newSet.delete(key);
      } else {
        newSet.add(key);
      }
      return newSet;
    });
  };


  // Group outputs by parent object
  const groupedOutputs = Object.entries(properties).reduce((acc, [key, property]) => {
    const parentKey = getParentKey(key) || key;
    if (!acc[parentKey]) {
      acc[parentKey] = [];
    }
    acc[parentKey].push([key, property]);
    return acc;
  }, {} as Record<string, [string, any][]>);

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

      <div className="flex flex-col items-end gap-2">
        {Object.entries(groupedOutputs).map(([parentKey, outputs]) => {
          const isParentConnected = isOutputConnected(nodeId, parentKey);
          const isParentExpanded = expandedObjects.has(parentKey);
          const shouldShowParent = isParentConnected || isOutputVisible;
          
          return (
            <div key={parentKey} className="flex flex-col items-end gap-1">
              {/* Parent object output */}
              {shouldShowParent && (
                <div className="relative flex items-center gap-2">
                  <Text
                    variant="body"
                    className="flex items-center gap-2 font-medium text-slate-700"
                  >
                    {outputs[0][1]?.description && (
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
                          <TooltipContent>{outputs[0][1]?.description}</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                    {outputs[0][1]?.title || parentKey}{" "}
                    <Text variant="small" as="span" className={getTypeDisplayInfo(outputs[0][1]).colorClass}>
                      ({getTypeDisplayInfo(outputs[0][1]).displayType})
                    </Text>
                  </Text>
                  <NodeHandle id={parentKey} isConnected={isParentConnected} side="right" />
                  
                  {/* Expand/collapse button for objects with sub-outputs */}
                  {outputs.length > 1 && (
                    <Button
                      variant="ghost"
                      className="p-1 h-6 w-6"
                      onClick={() => toggleObjectExpansion(parentKey)}
                    >
                      <CaretDownIcon
                        size={12}
                        weight="bold"
                        className={`transition-transform ${isParentExpanded ? "rotate-180" : ""}`}
                      />
                    </Button>
                  )}
                </div>
              )}
              
              {/* Sub-outputs */}
              {shouldShowParent && isParentExpanded && outputs.slice(1).map(([key, property]) => {
                const isConnected = isOutputConnected(nodeId, key);
                const { displayType, colorClass } = getTypeDisplayInfo(property);
                
                return (
                  <div key={key} className="relative flex items-center gap-2 ml-4">
                    <Text
                      variant="body"
                      className="flex items-center gap-2 font-medium text-slate-600"
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
                      <Text variant="small" as="span" className={colorClass}>
                        ({displayType})
                      </Text>
                    </Text>
                    <NodeHandle id={key} isConnected={isConnected} side="right" />
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
};
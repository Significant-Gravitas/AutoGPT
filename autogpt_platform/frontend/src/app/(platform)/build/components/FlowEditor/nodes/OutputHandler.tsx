import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CaretDownIcon, CaretRightIcon, InfoIcon } from "@phosphor-icons/react";
import { RJSFSchema } from "@rjsf/utils";
import { useState } from "react";

import { OutputNodeHandle } from "../handlers/NodeHandle";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { getTypeDisplayInfo } from "./helpers";
import { BlockUIType } from "../../types";
import { cn } from "@/lib/utils";
import { useBrokenOutputs } from "./useBrokenOutputs";

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
  const [expandedObjects, setExpandedObjects] = useState<
    Record<string, boolean>
  >({});

  const showHandles = uiType !== BlockUIType.OUTPUT;

  function toggleObjectExpanded(key: string) {
    setExpandedObjects((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  function hasConnectedOrBrokenDescendant(
    schema: RJSFSchema,
    keyPrefix: string,
  ): boolean {
    if (!schema) return false;
    return Object.entries(schema).some(
      ([key, fieldSchema]: [string, RJSFSchema]) => {
        const fullKey = keyPrefix ? `${keyPrefix}_#_${key}` : key;
        if (isOutputConnected(nodeId, fullKey) || brokenOutputs.has(fullKey))
          return true;
        if (fieldSchema?.properties)
          return hasConnectedOrBrokenDescendant(
            fieldSchema.properties,
            fullKey,
          );
        return false;
      },
    );
  }

  const renderOutputHandles = (
    schema: RJSFSchema,
    keyPrefix: string = "",
    titlePrefix: string = "",
    connectedOnly: boolean = false,
  ): React.ReactNode[] => {
    return Object.entries(schema).map(
      ([key, fieldSchema]: [string, RJSFSchema]) => {
        const fullKey = keyPrefix ? `${keyPrefix}_#_${key}` : key;
        const fieldTitle = titlePrefix + (fieldSchema?.title || key);

        const isConnected = isOutputConnected(nodeId, fullKey);
        const isBroken = brokenOutputs.has(fullKey);
        const hasNestedProperties = !!fieldSchema?.properties;
        const selfIsRelevant = isConnected || isBroken;
        const descendantIsRelevant =
          hasNestedProperties &&
          hasConnectedOrBrokenDescendant(fieldSchema.properties!, fullKey);

        const shouldShow = connectedOnly
          ? selfIsRelevant || descendantIsRelevant
          : isOutputVisible || selfIsRelevant || descendantIsRelevant;

        const { displayType, colorClass, hexColor } =
          getTypeDisplayInfo(fieldSchema);
        const isExpanded = expandedObjects[fullKey] ?? false;

        // User expanded → show all children; auto-expanded → filter to connected only
        const shouldRenderChildren = isExpanded || descendantIsRelevant;

        return shouldShow ? (
          <div
            key={fullKey}
            className="flex flex-col items-end gap-2"
            data-tutorial-id={`output-handler-${nodeId}-${fieldTitle}`}
          >
            <div className="relative flex items-center gap-2">
              {hasNestedProperties && (
                <button
                  onClick={() => toggleObjectExpanded(fullKey)}
                  className="flex items-center text-slate-500 hover:text-slate-700"
                  aria-label={isExpanded ? "Collapse" : "Expand"}
                >
                  {isExpanded ? (
                    <CaretDownIcon size={12} weight="bold" />
                  ) : (
                    <CaretRightIcon size={12} weight="bold" />
                  )}
                </button>
              )}
              {fieldSchema?.description && (
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
                    <TooltipContent>{fieldSchema?.description}</TooltipContent>
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
                {fieldTitle}
              </Text>
              <Text
                variant="small"
                as="span"
                className={cn(
                  colorClass,
                  isBroken && "!text-red-500 line-through",
                )}
              >
                ({displayType})
              </Text>

              {showHandles && (
                <OutputNodeHandle
                  isBroken={isBroken}
                  field_name={fullKey}
                  nodeId={nodeId}
                  hexColor={hexColor}
                />
              )}
            </div>

            {/* Nested properties */}
            {hasNestedProperties &&
              shouldRenderChildren &&
              renderOutputHandles(
                fieldSchema.properties!,
                fullKey,
                "",
                !isExpanded,
              )}
          </div>
        ) : null;
      },
    );
  };

  return (
    <div className="flex flex-col items-end justify-between gap-2 rounded-b-xlarge border-t border-zinc-200 bg-white py-3.5">
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

      <div className="flex flex-col items-end gap-2">
        {renderOutputHandles(properties, "", "", !isOutputVisible)}
      </div>
    </div>
  );
};

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
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { getTypeDisplayInfo } from "./helpers";
import { BlockUIType } from "../../types";
import { cn } from "@/lib/utils";
import { useBrokenOutputs } from "./useBrokenOutputs";
import {
  buildVisibleOutputTree,
  VisibleOutputTreeNode,
} from "./output-visibility";

export const OutputHandler = ({
  outputSchema,
  nodeId,
  uiType,
}: {
  outputSchema: RJSFSchema;
  nodeId: string;
  uiType: BlockUIType;
}) => {
  const edges = useEdgeStore((state) => state.edges);
  const nodeOutputCollapsedStates = useNodeStore(
    (state) => state.nodeOutputCollapsedStates,
  );
  const getOutputCollapsed = useNodeStore((state) => state.getOutputCollapsed);
  const toggleOutputCollapsed = useNodeStore(
    (state) => state.toggleOutputCollapsed,
  );
  const properties = (outputSchema?.properties || {}) as Record<
    string,
    RJSFSchema
  >;
  const [isOutputVisible, setIsOutputVisible] = useState(true);
  const brokenOutputs = useBrokenOutputs(nodeId);

  const showHandles = uiType !== BlockUIType.OUTPUT;
  const connectedSourceHandles = new Set(
    edges
      .filter((edge) => edge.source === nodeId && edge.sourceHandle)
      .map((edge) => edge.sourceHandle as string),
  );

  function isHandleConnected(handleId: string): boolean {
    return connectedSourceHandles.has(handleId);
  }

  function isOutputCollapsed(handleId: string): boolean {
    const collapseStateForNode = nodeOutputCollapsedStates[nodeId];
    if (
      collapseStateForNode &&
      Object.prototype.hasOwnProperty.call(collapseStateForNode, handleId)
    ) {
      return collapseStateForNode[handleId];
    }

    return getOutputCollapsed(nodeId, handleId);
  }

  const outputTree = buildVisibleOutputTree({
    properties,
    isHandleConnected,
    isCollapsed: isOutputCollapsed,
  });

  function renderOutputNode(node: VisibleOutputTreeNode): React.ReactNode {
    const shouldShowNode =
      isOutputVisible || node.isConnected || node.hasConnectedDescendant;
    if (!shouldShowNode) {
      return null;
    }

    const { displayType, colorClass, hexColor } = getTypeDisplayInfo(
      node.schema,
    );
    const isBroken = brokenOutputs.has(node.fullKey);
    const canToggleChildren = node.isObject && node.totalChildrenCount > 0;
    const rowIsCollapsed = isOutputCollapsed(node.fullKey);

    return (
      <div
        key={node.fullKey}
        className="flex flex-col items-end gap-2"
        data-tutorial-id={`output-handler-${nodeId}-${node.title}`}
      >
        <div className="relative flex items-center gap-2">
          {node.schema?.description && (
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
                <TooltipContent>{node.schema?.description}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}

          {canToggleChildren && (
            <button
              type="button"
              aria-label={`${rowIsCollapsed ? "Expand" : "Collapse"} ${node.title}`}
              aria-expanded={!rowIsCollapsed}
              className="flex items-center justify-center rounded-sm p-0.5 text-slate-500 hover:bg-zinc-100"
              onClick={() => toggleOutputCollapsed(nodeId, node.fullKey)}
            >
              {rowIsCollapsed ? (
                <CaretRightIcon size={12} weight="bold" />
              ) : (
                <CaretDownIcon size={12} weight="bold" />
              )}
            </button>
          )}

          <Text
            variant="body"
            className={cn(
              "text-slate-700",
              isBroken && "text-red-500 line-through",
            )}
          >
            {node.title}
          </Text>
          <Text
            variant="small"
            as="span"
            className={cn(colorClass, isBroken && "!text-red-500 line-through")}
          >
            ({displayType})
          </Text>

          {canToggleChildren &&
            rowIsCollapsed &&
            node.hiddenChildrenCount > 0 && (
              <Text variant="small" as="span" className="text-slate-500">
                ({node.hiddenChildrenCount})
              </Text>
            )}

          {showHandles && (
            <OutputNodeHandle
              isBroken={isBroken}
              field_name={node.fullKey}
              nodeId={nodeId}
              hexColor={hexColor}
            />
          )}
        </div>

        {node.children.map((child) => renderOutputNode(child))}
      </div>
    );
  }

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
        {outputTree.map((node) => renderOutputNode(node))}
      </div>
    </div>
  );
};

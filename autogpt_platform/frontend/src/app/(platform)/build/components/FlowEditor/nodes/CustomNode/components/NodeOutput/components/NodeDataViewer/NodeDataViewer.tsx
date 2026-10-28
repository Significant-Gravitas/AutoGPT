import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import {
  OutputActions,
  OutputItem,
} from "@/components/contextual/OutputRenderers";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { beautifyString } from "@/lib/utils";
import React, { FC } from "react";
import { useNodeDataViewer } from "./useNodeDataViewer";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useShallow } from "zustand/react/shallow";
import { NodeDataType } from "../../helpers";
import {
  ArrowExpand01Icon,
  Copy01Icon,
  Download04Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export interface NodeDataViewerProps {
  data?: any;
  pinName: string;
  nodeId?: string;
  execId?: string;
  isViewMoreData?: boolean;
  dataType?: NodeDataType;
}

export const NodeDataViewer: FC<NodeDataViewerProps> = ({
  data,
  pinName,
  nodeId,
  execId = "N/A",
  isViewMoreData = false,
  dataType = "output",
}) => {
  const executionResults = useNodeStore(
    useShallow((state) =>
      nodeId ? state.getNodeExecutionResults(nodeId) : [],
    ),
  );
  const latestInputData = useNodeStore(
    useShallow((state) =>
      nodeId ? state.getLatestNodeInputData(nodeId) : undefined,
    ),
  );
  const accumulatedOutputData = useNodeStore(
    useShallow((state) =>
      nodeId ? state.getAccumulatedNodeOutputData(nodeId) : {},
    ),
  );

  const resolvedData =
    data ??
    (dataType === "input"
      ? (latestInputData ?? {})
      : (accumulatedOutputData[pinName] ?? []));

  const {
    outputItems,
    copyExecutionId,
    handleCopyItem,
    handleDownloadItem,
    dataArray,
    copiedIndex,
    groupedExecutions,
    totalGroupedItems,
    handleCopyGroupedItem,
    handleDownloadGroupedItem,
    copiedKey,
  } = useNodeDataViewer(
    resolvedData,
    pinName,
    execId,
    executionResults,
    dataType,
  );

  const shouldGroupExecutions = groupedExecutions.length > 0;
  return (
    <Dialog title="Data Preview" styling={{ width: "600px" }}>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Dialog.Trigger>
              <Button
                variant="secondary"
                size="small"
                // TODO(tokens): no exact role: hover:text-slate-900 #2A313A
                className="h-fit min-w-0 gap-1.5 border border-border-base p-2 text-foreground-base hover:text-slate-900"
              >
                <Icon
                  icon={ArrowExpand01Icon}
                  size={isViewMoreData ? 16 : 12}
                />
              </Button>
            </Dialog.Trigger>
          </TooltipTrigger>
          <TooltipContent>
            <p>View Data</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Text
                // TODO(tokens): no exact role: text-slate-900 #2A313A
                variant="large-medium"
                className="text-slate-900"
              >
                Full {dataType === "input" ? "Input" : "Output"} Preview
              </Text>
            </div>
            <Text
              variant="small-medium-tight"
              as="div"
              // TODO(tokens): no exact role: border-slate-300 #B8BFCA, bg-slate-100 #EFF1F4
              className="rounded-full border border-slate-300 bg-slate-100 px-3 py-1.5"
            >
              {shouldGroupExecutions ? totalGroupedItems : dataArray.length}{" "}
              item
              {shouldGroupExecutions
                ? totalGroupedItems !== 1
                  ? "s"
                  : ""
                : dataArray.length !== 1
                  ? "s"
                  : ""}{" "}
              total
            </Text>
          </div>
          <Text
            variant="body-tight"
            as="div"
            // TODO(tokens): no exact role: text-gray-600 #4B5563
            className="text-gray-600"
          >
            {shouldGroupExecutions ? (
              <div>
                Pin:{" "}
                <Text variant="body-semibold-tight" as="span" tone="inherit">
                  {beautifyString(pinName)}
                </Text>
              </div>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <Text
                    // TODO(tokens): no exact role: text-slate-600 #5B6A7E
                    variant="body"
                    className="text-slate-600"
                  >
                    Execution ID:
                  </Text>
                  <Text
                    variant="small-medium-mono-tight"
                    as="p"
                    // TODO(tokens): no exact role: border-gray-300 #D1D5DB, bg-gray-50 #F9FAFB
                    className="rounded-full border border-gray-300 bg-gray-50 px-2 py-1"
                  >
                    {execId}
                  </Text>
                  <Button
                    variant="ghost"
                    size="small"
                    onClick={copyExecutionId}
                    className="h-6 w-6 min-w-0 p-0"
                  >
                    <Icon icon={Copy01Icon} size={14} />
                  </Button>
                </div>
                <div className="mt-2">
                  Pin:{" "}
                  <Text variant="body-semibold-tight" as="span" tone="inherit">
                    {beautifyString(pinName)}
                  </Text>
                </div>
              </>
            )}
          </Text>
        </div>

        <div className="flex-1">
          <div className="my-4">
            {shouldGroupExecutions ? (
              <div className="space-y-4">
                {groupedExecutions.map((execution) => (
                  <div
                    key={execution.execId}
                    // TODO(tokens): no fitting role for border-slate-200 #CFD4DB; #CFD4DB is node-border (the builder node ring), node-selected (the selected builder node ring)
                    className="rounded-3xl border border-slate-200 bg-surface p-4 shadow-sm"
                  >
                    <div className="flex items-center gap-2">
                      <Text
                        // TODO(tokens): no exact role: text-slate-600 #5B6A7E
                        variant="body"
                        className="text-slate-600"
                      >
                        Execution ID:
                      </Text>
                      <Text
                        variant="small-medium-mono-tight"
                        as="p"
                        // TODO(tokens): no exact role: border-gray-300 #D1D5DB, bg-gray-50 #F9FAFB
                        className="rounded-full border border-gray-300 bg-gray-50 px-2 py-1"
                      >
                        {execution.execId}
                      </Text>
                    </div>
                    <div className="mt-2 space-y-4">
                      {execution.outputItems.length > 0 ? (
                        execution.outputItems.map((item, index) => (
                          <div key={item.key} className="group">
                            <OutputItem
                              value={item.value}
                              metadata={item.metadata}
                              renderer={item.renderer}
                            />
                            <div className="mt-2 flex gap-3">
                              <Button
                                variant="secondary"
                                className="min-w-0 p-1"
                                size="icon"
                                onClick={() =>
                                  handleCopyGroupedItem(
                                    execution.execId,
                                    index,
                                    item,
                                  )
                                }
                                aria-label="Copy item"
                              >
                                {copiedKey ===
                                `${execution.execId}-${index}` ? (
                                  <Icon
                                    icon={Tick02Icon}
                                    className="size-4 text-success"
                                  />
                                ) : (
                                  <Icon
                                    icon={Copy01Icon}
                                    className="size-4 text-foreground-base"
                                  />
                                )}
                              </Button>
                              <Button
                                variant="secondary"
                                size="icon"
                                className="min-w-0 p-1"
                                onClick={() => handleDownloadGroupedItem(item)}
                                aria-label="Download item"
                              >
                                <Icon
                                  icon={Download04Icon}
                                  className="size-4 text-foreground-base"
                                />
                              </Button>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div
                          // TODO(tokens): no exact role: text-gray-500 #6B7280
                          className="py-4 text-center text-gray-500"
                        >
                          No data available
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : dataArray.length > 0 ? (
              <div className="space-y-4">
                {outputItems.map((item, index) => (
                  <div key={item.key} className="group">
                    <OutputItem
                      value={item.value}
                      metadata={item.metadata}
                      renderer={item.renderer}
                    />
                    <div className="mt-2 flex gap-3">
                      <Button
                        variant="secondary"
                        className="min-w-0 p-1"
                        size="icon"
                        onClick={() => handleCopyItem(index)}
                        aria-label="Copy item"
                      >
                        {copiedIndex === index ? (
                          <Icon
                            icon={Tick02Icon}
                            className="size-4 text-success"
                          />
                        ) : (
                          <Icon
                            icon={Copy01Icon}
                            className="size-4 text-foreground-base"
                          />
                        )}
                      </Button>
                      <Button
                        variant="secondary"
                        size="icon"
                        className="min-w-0 p-1"
                        onClick={() => handleDownloadItem(index)}
                        aria-label="Download item"
                      >
                        <Icon
                          icon={Download04Icon}
                          className="size-4 text-foreground-base"
                        />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div
                // TODO(tokens): no exact role: text-gray-500 #6B7280
                className="py-8 text-center text-gray-500"
              >
                No data available
              </div>
            )}
          </div>
        </div>

        <div className="flex justify-end pt-4">
          {outputItems.length > 1 && (
            <OutputActions
              items={outputItems.map((item) => ({
                value: item.value,
                metadata: item.metadata,
                renderer: item.renderer,
              }))}
              isPrimary
            />
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
};

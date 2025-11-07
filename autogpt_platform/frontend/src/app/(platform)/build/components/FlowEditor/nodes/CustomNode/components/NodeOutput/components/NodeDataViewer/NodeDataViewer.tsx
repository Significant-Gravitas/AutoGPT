import React, { FC } from "react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import {
  ArrowsOutSimpleIcon,
  CopyIcon,
  DownloadIcon,
  CheckIcon,
} from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { beautifyString } from "@/lib/utils";
import { ScrollArea } from "@/components/__legacy__/ui/scroll-area";
import {
  OutputItem,
  OutputActions,
} from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers";
import { useNodeDataViewer } from "./useNodeDataViewer";

interface NodeDataViewerProps {
  data: any;
  pinName: string;
  execId?: string;
  isViewMoreData?: boolean;
}

export const NodeDataViewer: FC<NodeDataViewerProps> = ({
  data,
  pinName,
  execId = "N/A",
  isViewMoreData = false,
}) => {
  const {
    outputItems,
    copyExecutionId,
    handleCopyItem,
    handleDownloadItem,
    dataArray,
    copiedIndex,
  } = useNodeDataViewer(data, pinName, execId);
  return (
    <Dialog styling={{ width: "600px" }}>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Dialog.Trigger>
              <Button
                variant="secondary"
                size="small"
                className="h-fit min-w-0 gap-1.5 border border-zinc-200 p-2 text-black hover:text-slate-900"
              >
                <ArrowsOutSimpleIcon size={isViewMoreData ? 16 : 12} />
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
              <Text variant="large-medium" className="text-slate-900">
                Full Output Preview
              </Text>
            </div>
            <div className="rounded-full border border-slate-300 bg-slate-100 px-3 py-1.5 text-xs font-medium text-black">
              {dataArray.length} item{dataArray.length !== 1 ? "s" : ""} total
            </div>
          </div>
          <div className="text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <Text variant="body" className="text-slate-600">
                Execution ID:
              </Text>
              <Text
                variant="body-medium"
                className="rounded-full border border-gray-300 bg-gray-50 px-2 py-1 font-mono text-xs"
              >
                {execId}
              </Text>
              <Button
                variant="ghost"
                size="small"
                onClick={copyExecutionId}
                className="h-6 w-6 min-w-0 p-0"
              >
                <CopyIcon size={14} />
              </Button>
            </div>
            <div className="mt-2">
              Pin:{" "}
              <span className="font-semibold">{beautifyString(pinName)}</span>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-hidden">
          <ScrollArea className="h-full">
            <div className="my-4">
              {dataArray.length > 0 ? (
                <div className="space-y-4">
                  {outputItems.map((item, index) => (
                    <div key={item.key} className="group relative">
                      <OutputItem
                        value={item.value}
                        metadata={item.metadata}
                        renderer={item.renderer}
                      />
                      <div className="absolute right-3 top-3 flex gap-3">
                        <Button
                          variant="secondary"
                          className="min-w-0 p-1"
                          size="icon"
                          onClick={() => handleCopyItem(index)}
                          aria-label="Copy item"
                        >
                          {copiedIndex === index ? (
                            <CheckIcon className="size-4 text-green-600" />
                          ) : (
                            <CopyIcon className="size-4 text-black" />
                          )}
                        </Button>
                        <Button
                          variant="secondary"
                          size="icon"
                          className="min-w-0 p-1"
                          onClick={() => handleDownloadItem(index)}
                          aria-label="Download item"
                        >
                          <DownloadIcon className="size-4 text-black" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-8 text-center text-gray-500">
                  No data available
                </div>
              )}
            </div>
          </ScrollArea>
        </div>

        <div className="flex justify-end pt-4">
          {outputItems.length > 0 && (
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

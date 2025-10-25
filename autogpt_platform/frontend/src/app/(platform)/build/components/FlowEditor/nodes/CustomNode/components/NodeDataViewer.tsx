import React, { FC, useMemo, useState } from "react";
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
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ScrollArea } from "@/components/__legacy__/ui/scroll-area";
import {
  globalRegistry,
  OutputItem,
  OutputActions,
} from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers";
import type { OutputMetadata } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers";
import { downloadOutputs } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers/utils/download";

interface NodeDataViewerProps {
  data: any;
  pinName: string;
  execId?: string;
}

export const NodeDataViewer: FC<NodeDataViewerProps> = ({
  data,
  pinName,
  execId = "N/A",
}) => {
  const { toast } = useToast();
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  // Normalize data to array format
  const dataArray = useMemo(() => {
    return Array.isArray(data) ? data : [data];
  }, [data]);

  // Prepare items for the enhanced renderer system
  const outputItems = useMemo(() => {
    if (!dataArray) return [];

    const items: Array<{
      key: string;
      label: string;
      value: unknown;
      metadata?: OutputMetadata;
      renderer: any;
    }> = [];

    dataArray.forEach((value, index) => {
      const metadata: OutputMetadata = {};

      // Extract metadata from the value if it's an object
      if (
        typeof value === "object" &&
        value !== null &&
        !React.isValidElement(value)
      ) {
        const objValue = value as any;
        if (objValue.type) metadata.type = objValue.type;
        if (objValue.mimeType) metadata.mimeType = objValue.mimeType;
        if (objValue.filename) metadata.filename = objValue.filename;
        if (objValue.language) metadata.language = objValue.language;
      }

      const renderer = globalRegistry.getRenderer(value, metadata);
      if (renderer) {
        items.push({
          key: `item-${index}`,
          label: index === 0 ? beautifyString(pinName) : "",
          value,
          metadata,
          renderer,
        });
      } else {
        // Fallback to text renderer
        const textRenderer = globalRegistry
          .getAllRenderers()
          .find((r) => r.name === "TextRenderer");
        if (textRenderer) {
          items.push({
            key: `item-${index}`,
            label: index === 0 ? beautifyString(pinName) : "",
            value:
              typeof value === "string"
                ? value
                : JSON.stringify(value, null, 2),
            metadata,
            renderer: textRenderer,
          });
        }
      }
    });

    return items;
  }, [dataArray, pinName]);

  const copyExecutionId = () => {
    navigator.clipboard.writeText(execId).then(() => {
      toast({
        title: "Execution ID copied to clipboard!",
        duration: 2000,
      });
    });
  };

  const handleCopyItem = async (index: number) => {
    const item = outputItems[index];
    const copyContent = item.renderer.getCopyContent(item.value, item.metadata);

    if (copyContent) {
      try {
        let text: string;
        if (typeof copyContent.data === "string") {
          text = copyContent.data;
        } else if (copyContent.fallbackText) {
          text = copyContent.fallbackText;
        } else {
          return;
        }

        await navigator.clipboard.writeText(text);
        setCopiedIndex(index);
        setTimeout(() => setCopiedIndex(null), 2000);
      } catch (error) {
        console.error("Failed to copy:", error);
      }
    }
  };

  const handleDownloadItem = (index: number) => {
    const item = outputItems[index];
    downloadOutputs([
      {
        value: item.value,
        metadata: item.metadata,
        renderer: item.renderer,
      },
    ]);
  };

  return (
    <Dialog>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Dialog.Trigger>
              <Button
                variant="secondary"
                size="small"
                className="h-fit min-w-0 gap-1.5 p-2 text-black hover:text-slate-900"
              >
                <ArrowsOutSimpleIcon size={12} />
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
              <span>Execution ID:</span>
              <span className="rounded-full border border-gray-300 bg-gray-50 px-2 py-1 font-mono text-xs">
                {execId}
              </span>
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
            <div className="my-4 border-l-2 border-slate-50 pl-4">
              {dataArray.length > 0 ? (
                <div className="space-y-4">
                  {outputItems.map((item, index) => (
                    <div key={item.key} className="group relative">
                      <OutputItem
                        value={item.value}
                        metadata={item.metadata}
                        renderer={item.renderer}
                      />
                      <div className="absolute right-4 top-4 flex gap-3">
                        <Button
                          variant="ghost"
                          className="min-w-0 p-0"
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
                          variant="ghost"
                          size="icon"
                          className="min-w-0 p-0"
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

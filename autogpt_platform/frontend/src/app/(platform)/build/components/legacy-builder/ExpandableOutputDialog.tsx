import type { OutputMetadata } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/OutputRenderers";
import {
  globalRegistry,
  OutputActions,
  OutputItem,
} from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/OutputRenderers";
import { beautifyString } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { Clipboard, Maximize2 } from "lucide-react";
import React, { FC, useMemo, useState } from "react";
import { Button } from "../../../../../components/__legacy__/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../../../../../components/__legacy__/ui/dialog";
import { ContentRenderer } from "../../../../../components/__legacy__/ui/render";
import { ScrollArea } from "../../../../../components/__legacy__/ui/scroll-area";
import { Separator } from "../../../../../components/__legacy__/ui/separator";
import { Switch } from "../../../../../components/atoms/Switch/Switch";
import { useToast } from "../../../../../components/molecules/Toast/use-toast";

interface ExpandableOutputDialogProps {
  isOpen: boolean;
  onClose: () => void;
  execId: string;
  pinName: string;
  data: any[];
}

const ExpandableOutputDialog: FC<ExpandableOutputDialogProps> = ({
  isOpen,
  onClose,
  execId,
  pinName,
  data,
}) => {
  const { toast } = useToast();
  const enableEnhancedOutputHandling = useGetFlag(
    Flag.ENABLE_ENHANCED_OUTPUT_HANDLING,
  );
  const [useEnhancedRenderer, setUseEnhancedRenderer] = useState(false);

  // Prepare items for the enhanced renderer system
  const outputItems = useMemo(() => {
    if (!data || !useEnhancedRenderer) return [];

    const items: Array<{
      key: string;
      label: string;
      value: unknown;
      metadata?: OutputMetadata;
      renderer: any;
    }> = [];

    data.forEach((value, index) => {
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
  }, [data, useEnhancedRenderer, pinName]);

  const copyData = () => {
    const formattedData = data
      .map((item) =>
        typeof item === "object" ? JSON.stringify(item, null, 2) : String(item),
      )
      .join("\n\n");

    navigator.clipboard.writeText(formattedData).then(() => {
      toast({
        title: `"${beautifyString(pinName)}" data copied to clipboard!`,
        duration: 2000,
      });
    });
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="flex h-[90vh] w-[90vw] max-w-4xl flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between pr-8">
            <div className="flex items-center gap-2">
              <Maximize2 size={20} />
              Full Output Preview
            </div>
            {enableEnhancedOutputHandling && (
              <div className="flex items-center gap-3">
                <label
                  htmlFor="enhanced-rendering-toggle"
                  className="cursor-pointer select-none text-sm font-normal text-gray-600"
                >
                  Enhanced Rendering
                </label>
                <Switch
                  id="enhanced-rendering-toggle"
                  checked={useEnhancedRenderer}
                  onCheckedChange={setUseEnhancedRenderer}
                />
              </div>
            )}
          </DialogTitle>
          <DialogDescription>
            Execution ID: <span className="font-mono text-xs">{execId}</span>
            <br />
            Pin:{" "}
            <span className="font-semibold">{beautifyString(pinName)}</span>
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-hidden">
          {useEnhancedRenderer && outputItems.length > 0 && (
            <div className="border-b px-4 py-2">
              <OutputActions
                items={outputItems.map((item) => ({
                  value: item.value,
                  metadata: item.metadata,
                  renderer: item.renderer,
                }))}
              />
            </div>
          )}
          <ScrollArea className="h-full">
            <div className="p-4">
              {data.length > 0 ? (
                useEnhancedRenderer ? (
                  <div className="space-y-4">
                    {outputItems.map((item) => (
                      <OutputItem
                        key={item.key}
                        value={item.value}
                        metadata={item.metadata}
                        renderer={item.renderer}
                        label={item.label}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="space-y-4">
                    {data.map((item, index) => (
                      <div
                        key={index}
                        className="rounded-lg border bg-gray-50 p-4"
                      >
                        <div className="mb-2 flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-600">
                            Item {index + 1} of {data.length}
                          </span>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              const itemData =
                                typeof item === "object"
                                  ? JSON.stringify(item, null, 2)
                                  : String(item);
                              navigator.clipboard
                                .writeText(itemData)
                                .then(() => {
                                  toast({
                                    title: `Item ${index + 1} copied to clipboard!`,
                                    duration: 2000,
                                  });
                                });
                            }}
                            className="flex items-center gap-1"
                          >
                            <Clipboard size={14} />
                            Copy Item
                          </Button>
                        </div>
                        <Separator className="mb-3" />
                        <div className="whitespace-pre-wrap break-words font-mono text-sm">
                          <ContentRenderer
                            value={item}
                            truncateLongData={false}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )
              ) : (
                <div className="py-8 text-center text-gray-500">
                  No data available
                </div>
              )}
            </div>
          </ScrollArea>
        </div>

        <DialogFooter className="flex justify-between">
          <div className="text-sm text-gray-600">
            {data.length} item{data.length !== 1 ? "s" : ""} total
          </div>
          <div className="flex gap-2">
            {!useEnhancedRenderer && (
              <Button
                variant="outline"
                onClick={copyData}
                className="flex items-center gap-1"
              >
                <Clipboard size={16} />
                Copy All
              </Button>
            )}
            <Button onClick={onClose}>Close</Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default ExpandableOutputDialog;

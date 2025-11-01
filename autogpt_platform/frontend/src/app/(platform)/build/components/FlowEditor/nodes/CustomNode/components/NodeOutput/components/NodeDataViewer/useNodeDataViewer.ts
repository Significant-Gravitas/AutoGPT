import React, { useState, useMemo } from "react";
import type { OutputMetadata } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers";
import { downloadOutputs } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers/utils/download";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { globalRegistry } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OutputRenderers";
import { beautifyString } from "@/lib/utils";

export const useNodeDataViewer = (
  data: any,
  pinName: string,
  execId: string,
) => {
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

  return {
    outputItems,
    dataArray,
    copyExecutionId,
    handleCopyItem,
    handleDownloadItem,
    copiedIndex,
  };
};

import { downloadOutputs } from "@/components/contextual/OutputRenderers/utils/download";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { beautifyString } from "@/lib/utils";
import { useState } from "react";
import type { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import {
  NodeDataType,
  createOutputItems,
  getExecutionData,
  normalizeToArray,
  type OutputItem,
} from "../../helpers";

export type GroupedExecution = {
  execId: string;
  outputItems: Array<OutputItem>;
};

export const useNodeDataViewer = (
  data: any,
  pinName: string,
  execId: string,
  executionResults?: NodeExecutionResult[],
  dataType?: NodeDataType,
) => {
  const { toast } = useToast();
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const dataArray = Array.isArray(data) ? data : [data];

  const outputItems =
    !dataArray || dataArray.length === 0
      ? []
      : createOutputItems(dataArray).map((item, index) => ({
          ...item,
          label: index === 0 ? beautifyString(pinName) : "",
        }));

  const groupedExecutions =
    !executionResults || executionResults.length === 0
      ? []
      : [...executionResults].reverse().map((result) => {
          const rawData = getExecutionData(
            result,
            dataType || "output",
            pinName,
          );
          let dataArray: unknown[];
          if (dataType === "input") {
            dataArray =
              rawData !== undefined && rawData !== null ? [rawData] : [];
          } else {
            dataArray = normalizeToArray(rawData);
          }

          const outputItems = createOutputItems(dataArray);
          return {
            execId: result.node_exec_id,
            outputItems,
          };
        });

  const totalGroupedItems = groupedExecutions.reduce(
    (total, execution) => total + execution.outputItems.length,
    0,
  );

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

  const handleCopyGroupedItem = async (
    execId: string,
    index: number,
    item: OutputItem,
  ) => {
    const copyContent = item.renderer.getCopyContent(item.value, item.metadata);

    if (!copyContent) {
      return;
    }

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
      setCopiedKey(`${execId}-${index}`);
      setTimeout(() => setCopiedKey(null), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  const handleDownloadGroupedItem = (item: OutputItem) => {
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
    groupedExecutions,
    totalGroupedItems,
    handleCopyGroupedItem,
    handleDownloadGroupedItem,
    copiedKey,
  };
};

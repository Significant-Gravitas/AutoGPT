import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useShallow } from "zustand/react/shallow";
import { useState } from "react";

export const useNodeOutput = (nodeId: string) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const { toast } = useToast();

  const nodeExecutionResult = useNodeStore(
    useShallow((state) => state.getNodeExecutionResult(nodeId)),
  );

  const inputData = nodeExecutionResult?.input_data;

  const outputData: Record<string, Array<any>> = {
    ...nodeExecutionResult?.output_data,
  };
  const handleCopy = async (key: string, value: any) => {
    try {
      const text = JSON.stringify(value, null, 2);
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      toast({
        title: "Copied to clipboard!",
        duration: 2000,
      });
      setTimeout(() => setCopiedKey(null), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
      toast({
        title: "Failed to copy",
        variant: "destructive",
        duration: 2000,
      });
    }
  };
  return {
    outputData: outputData,
    inputData: inputData,
    isExpanded: isExpanded,
    setIsExpanded: setIsExpanded,
    copiedKey: copiedKey,
    setCopiedKey: setCopiedKey,
    handleCopy: handleCopy,
    executionResultId: nodeExecutionResult?.node_exec_id,
  };
};

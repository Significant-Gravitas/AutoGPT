import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { beautifyString, cn } from "@/lib/utils";
import { ContentRenderer } from "./ContentRenderer";
import { ScrollArea } from "@/components/__legacy__/ui/scroll-area";
import { useState } from "react";
import { NodeDataViewer } from "./NodeDataViewer/NodeDataViewer";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { CheckIcon, CopyIcon } from "@phosphor-icons/react";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useShallow } from "zustand/react/shallow";
import {
  NodeDataType,
  getExecutionEntries,
  normalizeToArray,
} from "../helpers";

export const ViewMoreData = ({
  nodeId,
  dataType = "output",
}: {
  nodeId: string;
  dataType?: NodeDataType;
}) => {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const { toast } = useToast();
  const executionResults = useNodeStore(
    useShallow((state) => state.getNodeExecutionResults(nodeId)),
  );

  const reversedExecutionResults = [...executionResults].reverse();

  const handleCopy = (key: string, value: any) => {
    const textToCopy =
      typeof value === "object"
        ? JSON.stringify(value, null, 2)
        : String(value);
    navigator.clipboard.writeText(textToCopy);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const copyExecutionId = (executionId: string) => {
    navigator.clipboard.writeText(executionId || "N/A").then(() => {
      toast({
        title: "Execution ID copied to clipboard!",
        duration: 2000,
      });
    });
  };

  return (
    <Dialog styling={{ width: "600px", paddingRight: "16px" }}>
      <Dialog.Trigger>
        <Button
          variant="secondary"
          size="small"
          className="h-fit w-fit min-w-0 !text-xs"
        >
          View More
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="h4" className="text-slate-900">
            Complete {dataType === "input" ? "Input" : "Output"} Data
          </Text>

          <ScrollArea className="h-full">
            <div className="flex flex-col gap-4">
              {reversedExecutionResults.map((result) => (
                <div
                  key={result.node_exec_id}
                  className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm"
                >
                  <div className="flex items-center gap-2">
                    <Text variant="body" className="text-slate-600">
                      Execution ID:
                    </Text>
                    <Text
                      variant="body-medium"
                      className="rounded-full border border-gray-300 bg-gray-50 px-2 py-1 font-mono text-xs"
                    >
                      {result.node_exec_id}
                    </Text>
                    <Button
                      variant="ghost"
                      size="small"
                      onClick={() => copyExecutionId(result.node_exec_id)}
                      className="h-6 w-6 min-w-0 p-0"
                    >
                      <CopyIcon size={14} />
                    </Button>
                  </div>

                  <div className="mt-4 flex flex-col gap-4">
                    {getExecutionEntries(result, dataType).map(
                      ([key, value]) => {
                        const normalizedValue = normalizeToArray(value);
                        return (
                          <div key={key} className="flex flex-col gap-2">
                            <div className="flex items-center gap-2">
                              <Text
                                variant="body-medium"
                                className="!font-semibold text-slate-600"
                              >
                                Pin:
                              </Text>
                              <Text
                                variant="body-medium"
                                className="text-slate-700"
                              >
                                {beautifyString(key)}
                              </Text>
                            </div>
                            <div className="w-full space-y-2">
                              <Text
                                variant="body-medium"
                                className="!font-semibold text-slate-600"
                              >
                                Data:
                              </Text>
                              <div className="relative space-y-2">
                                {normalizedValue.map((item, index) => (
                                  <div key={index}>
                                    <ContentRenderer
                                      value={item}
                                      shortContent={false}
                                    />
                                  </div>
                                ))}

                                <div className="mt-1 flex justify-end gap-1">
                                  <NodeDataViewer
                                    data={normalizedValue}
                                    pinName={key}
                                    execId={result.node_exec_id}
                                    isViewMoreData={true}
                                    dataType={dataType}
                                  />
                                  <Button
                                    variant="secondary"
                                    size="small"
                                    onClick={() =>
                                      handleCopy(
                                        `${result.node_exec_id}-${key}`,
                                        normalizedValue,
                                      )
                                    }
                                    className={cn(
                                      "h-fit min-w-0 gap-1.5 border border-zinc-200 p-2 text-black hover:text-slate-900",
                                      copiedKey ===
                                        `${result.node_exec_id}-${key}` &&
                                        "border-green-400 bg-green-100 hover:border-green-400 hover:bg-green-200",
                                    )}
                                  >
                                    {copiedKey ===
                                    `${result.node_exec_id}-${key}` ? (
                                      <CheckIcon
                                        size={16}
                                        className="text-green-600"
                                      />
                                    ) : (
                                      <CopyIcon size={16} />
                                    )}
                                  </Button>
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      },
                    )}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>
      </Dialog.Content>
    </Dialog>
  );
};

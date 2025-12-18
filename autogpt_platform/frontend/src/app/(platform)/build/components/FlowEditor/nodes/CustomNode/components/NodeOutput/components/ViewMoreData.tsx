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

export const ViewMoreData = ({
  outputData,
  execId,
}: {
  outputData: Record<string, Array<any>>;
  execId?: string;
}) => {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const { toast } = useToast();

  const handleCopy = (key: string, value: any) => {
    const textToCopy =
      typeof value === "object"
        ? JSON.stringify(value, null, 2)
        : String(value);
    navigator.clipboard.writeText(textToCopy);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const copyExecutionId = () => {
    navigator.clipboard.writeText(execId || "N/A").then(() => {
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
          variant="primary"
          size="small"
          className="h-fit w-fit min-w-0 !text-xs"
        >
          View More
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="h4" className="text-slate-900">
            Complete Output Data
          </Text>

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

          <ScrollArea className="h-full">
            <div className="flex flex-col gap-4">
              {Object.entries(outputData).map(([key, value]) => (
                <div key={key} className="flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <Text
                      variant="body-medium"
                      className="!font-semibold text-slate-600"
                    >
                      Pin:
                    </Text>
                    <Text variant="body-medium" className="text-slate-700">
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
                      {value.map((item, index) => (
                        <div key={index}>
                          <ContentRenderer value={item} shortContent={false} />
                        </div>
                      ))}

                      <div className="mt-1 flex justify-end gap-1">
                        <NodeDataViewer
                          data={value}
                          pinName={key}
                          execId={execId}
                          isViewMoreData={true}
                        />
                        <Button
                          variant="secondary"
                          size="small"
                          onClick={() => handleCopy(key, value)}
                          className={cn(
                            "h-fit min-w-0 gap-1.5 border border-zinc-200 p-2 text-black hover:text-slate-900",
                            copiedKey === key &&
                              "border-green-400 bg-green-100 hover:border-green-400 hover:bg-green-200",
                          )}
                        >
                          {copiedKey === key ? (
                            <CheckIcon size={16} className="text-green-600" />
                          ) : (
                            <CopyIcon size={16} />
                          )}
                        </Button>
                      </div>
                    </div>
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

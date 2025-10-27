import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { beautifyString, cn } from "@/lib/utils";
import {
  CaretDownIcon,
  CopyIcon,
  CheckIcon,
  CaretLineDownIcon,
} from "@phosphor-icons/react";
import { NodeDataViewer } from "./components/NodeDataViewer";
import { ContentRenderer } from "./components/ContentRenderer";
import { useNodeOutput } from "./useNodeOutput";
import { ViewMoreData } from "./components/ViewMoreData";
import { Separator } from "@/components/__legacy__/ui/separator";

export const NodeDataRenderer = ({ nodeId }: { nodeId: string }) => {
  const {
    outputData,
    isExpanded,
    setIsExpanded,
    copiedKey,
    handleCopy,
    executionResultId,
    inputData,
  } = useNodeOutput(nodeId);

  if (Object.keys(outputData).length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col gap-3 rounded-b-xl border-t border-slate-200/50 px-4 py-4">
      <div className="flex items-center justify-between">
        <Text variant="body-medium" className="!font-semibold text-slate-700">
          Node Output
        </Text>
        <Button
          variant="ghost"
          size="small"
          onClick={() => setIsExpanded(!isExpanded)}
          className="h-fit min-w-0 p-1 text-slate-600 hover:text-slate-900"
        >
          <CaretDownIcon
            size={16}
            weight="bold"
            className={`transition-transform ${isExpanded ? "rotate-180" : ""}`}
          />
        </Button>
      </div>

      {isExpanded && (
        <>
          <div className="flex max-w-[350px] flex-col gap-4">
            <div className="space-y-2">
              <div className="w-fit rounded-xlarge border border-purple-400 bg-purple-100 p-1 px-4">
                <Text
                  variant="small-medium"
                  className="flex items-center gap-1 !text-xs text-purple-700"
                >
                  Input
                  <CaretLineDownIcon size={12} weight="bold" className="ml-1" />
                </Text>
              </div>
              <div
                className={cn(
                  "rounded-xlarge border border-transparent from-zinc-100 to-white p-3 text-slate-700 [background-clip:padding-box,border-box] [background-image:linear-gradient(to_bottom_right,rgb(244_244_245),white),linear-gradient(to_bottom_right,rgb(228_228_231),rgb(250_250_250))] [background-origin:padding-box,border-box]",
                )}
              >
                <ContentRenderer value={inputData} truncateLongData={true} />
              </div>
              <div className="mt-1 flex justify-end gap-1">
                <NodeDataViewer
                  data={inputData}
                  pinName="Input"
                  execId={executionResultId}
                />
                <Button
                  variant="secondary"
                  size="small"
                  onClick={() => handleCopy("input", inputData)}
                  className={cn(
                    "h-fit min-w-0 gap-1.5 border border-zinc-200 p-2 text-black hover:text-slate-900",
                    copiedKey === "input" &&
                      "border-green-400 bg-green-100 hover:border-green-400 hover:bg-green-200",
                  )}
                >
                  {copiedKey === "input" ? (
                    <CheckIcon size={12} className="text-green-600" />
                  ) : (
                    <CopyIcon size={12} />
                  )}
                </Button>
              </div>
            </div>
            <Separator />
            {Object.entries(outputData)
              .slice(0, 2)
              .map(([key, value]) => (
                <div key={key} className="flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <Text
                      variant="small-medium"
                      className="!font-semibold text-slate-600"
                    >
                      Pin:
                    </Text>
                    <Text variant="small" className="text-slate-700">
                      {beautifyString(key)}
                    </Text>
                  </div>
                  <div className="w-full space-y-2">
                    <Text
                      variant="small"
                      className="!font-semibold text-slate-600"
                    >
                      Data:
                    </Text>
                    <div className="relative">
                      {value.map((item, index) => (
                        <div
                          key={index}
                          className={cn(
                            "rounded-xlarge border border-transparent from-zinc-100 to-white p-3 text-slate-700 [background-clip:padding-box,border-box] [background-image:linear-gradient(to_bottom_right,rgb(244_244_245),white),linear-gradient(to_bottom_right,rgb(228_228_231),rgb(250_250_250))] [background-origin:padding-box,border-box]",
                            index < value.length - 1 && "mb-2",
                          )}
                        >
                          <ContentRenderer
                            value={item}
                            truncateLongData={true}
                          />
                        </div>
                      ))}

                      <div className="mt-1 flex justify-end gap-1">
                        <NodeDataViewer
                          data={value}
                          pinName={key}
                          execId={executionResultId}
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
                            <CheckIcon size={12} className="text-green-600" />
                          ) : (
                            <CopyIcon size={12} />
                          )}
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
          </div>

          {/* TODO: Currently this button is not working, need to make it working */}
          {Object.keys(outputData).length > 2 && (
            <ViewMoreData outputData={outputData} execId={executionResultId} />
          )}
        </>
      )}
    </div>
  );
};

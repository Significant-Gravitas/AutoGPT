import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { beautifyString, cn } from "@/lib/utils";
import { CopyIcon, CheckIcon } from "@phosphor-icons/react";
import { NodeDataViewer } from "./components/NodeDataViewer/NodeDataViewer";
import { ContentRenderer } from "./components/ContentRenderer";
import { useNodeOutput } from "./useNodeOutput";
import { ViewMoreData } from "./components/ViewMoreData";

export const NodeDataRenderer = ({ nodeId }: { nodeId: string }) => {
  const {
    latestOutputData,
    copiedKey,
    handleCopy,
    executionResultId,
    latestInputData,
  } = useNodeOutput(nodeId);

  if (Object.keys(latestOutputData).length === 0) {
    return null;
  }

  return (
    <div
      data-tutorial-id={`node-output`}
      className="rounded-b-xl border-t border-zinc-200 px-4 py-2"
    >
      <Accordion type="single" collapsible defaultValue="node-output">
        <AccordionItem value="node-output" className="border-none">
          <AccordionTrigger className="py-2 hover:no-underline">
            <Text
              variant="body-medium"
              className="!font-semibold text-slate-700"
            >
              Node Output
            </Text>
          </AccordionTrigger>
          <AccordionContent className="pt-2">
            <div className="flex max-w-[350px] flex-col gap-4">
              <div className="space-y-2">
                <Text variant="small-medium">Input</Text>

                <ContentRenderer value={latestInputData} shortContent={false} />

                <div className="mt-1 flex justify-end gap-1">
                  <NodeDataViewer
                    pinName="Input"
                    nodeId={nodeId}
                    execId={executionResultId}
                    dataType="input"
                  />
                  <Button
                    variant="secondary"
                    size="small"
                    onClick={() => handleCopy("input", latestInputData)}
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

              {Object.entries(latestOutputData)
                .slice(0, 2)
                .map(([key, value]) => {
                  return (
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
                        <div className="relative space-y-2">
                          {value.map((item, index) => (
                            <div key={index}>
                              <ContentRenderer
                                value={item}
                                shortContent={true}
                              />
                            </div>
                          ))}

                          <div className="mt-1 flex justify-end gap-1">
                            <NodeDataViewer
                              pinName={key}
                              nodeId={nodeId}
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
                                <CheckIcon
                                  size={12}
                                  className="text-green-600"
                                />
                              ) : (
                                <CopyIcon size={12} />
                              )}
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
            </div>
            <ViewMoreData nodeId={nodeId} />
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
};

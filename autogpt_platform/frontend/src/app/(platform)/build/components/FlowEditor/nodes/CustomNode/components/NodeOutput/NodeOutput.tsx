import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { beautifyString } from "@/lib/utils";
import { NodeDataViewer } from "./components/NodeDataViewer/NodeDataViewer";
import { ContentRenderer } from "./components/ContentRenderer";
import { useNodeOutput } from "./useNodeOutput";
import { ViewMoreData } from "./components/ViewMoreData";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
      className="rounded-b-xl border-t border-border-base px-4 py-2"
    >
      <Accordion type="single" collapsible defaultValue="node-output">
        <AccordionItem value="node-output" className="border-none">
          <AccordionTrigger className="py-2 hover:no-underline">
            <Text variant="body-semibold" tone="body">
              Node Output
            </Text>
          </AccordionTrigger>
          <AccordionContent className="pt-2">
            <div className="flex max-w-[350px] flex-col gap-4">
              <div className="space-y-2">
                <Text variant="small-medium">Input</Text>

                <ContentRenderer value={latestInputData} shortContent={true} />

                <div className="mt-1 flex justify-end gap-1">
                  <NodeDataViewer
                    pinName="Input"
                    nodeId={nodeId}
                    execId={executionResultId}
                    dataType="input"
                  />
                  <Button
                    variant={copiedKey === "input" ? "success" : "secondary"}
                    size="icon-sm"
                    shape="pill"
                    onClick={() => handleCopy("input", latestInputData)}
                    tone="base"
                  >
                    {copiedKey === "input" ? (
                      <Icon icon={Tick02Icon} size={12} tone="success" />
                    ) : (
                      <Icon icon={Copy01Icon} size={12} />
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
                        <Text variant="small-semibold" tone="secondary">
                          Pin:
                        </Text>
                        <Text variant="small" tone="body">
                          {beautifyString(key)}
                        </Text>
                      </div>
                      <div className="w-full space-y-2">
                        <Text variant="small-semibold" tone="secondary">
                          Data:
                        </Text>
                        <div className="relative space-y-2">
                          {value.slice(0, 3).map((item, index) => (
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
                              variant={
                                copiedKey === key ? "success" : "secondary"
                              }
                              size="icon-sm"
                              shape="pill"
                              onClick={() => handleCopy(key, value)}
                              tone="base"
                            >
                              {copiedKey === key ? (
                                <Icon
                                  icon={Tick02Icon}
                                  size={12}
                                  tone="success"
                                />
                              ) : (
                                <Icon icon={Copy01Icon} size={12} />
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

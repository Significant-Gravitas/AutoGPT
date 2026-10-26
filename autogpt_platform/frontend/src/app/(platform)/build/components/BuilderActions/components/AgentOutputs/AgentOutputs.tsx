import { BlockUIType } from "@/app/(platform)/build/components/types";
import { Label } from "@/components/atoms/Label/Label";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { ScrollArea } from "@/components/molecules/ScrollArea/ScrollArea";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/molecules/Dialog/components/SheetPrimitives";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import {
  globalRegistry,
  OutputActions,
  OutputItem,
} from "@/components/contextual/OutputRenderers";
import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

export const AgentOutputs = ({ flowID }: { flowID: string | null }) => {
  const hasOutputs = useGraphStore(useShallow((state) => state.hasOutputs));
  const nodes = useNodeStore(useShallow((state) => state.nodes));

  const outputs = useMemo(() => {
    const outputNodes = nodes.filter(
      (node) => node.data.uiType === BlockUIType.OUTPUT,
    );

    return outputNodes
      .map((node) => {
        const executionResults = node.data.nodeExecutionResults || [];

        const items = executionResults
          .filter((result) => result.output_data?.output !== undefined)
          .map((result) => {
            const outputData = result.output_data!.output;
            const renderer = globalRegistry.getRenderer(outputData);
            return {
              nodeExecID: result.node_exec_id,
              value: outputData,
              renderer,
            };
          })
          .filter(
            (
              item,
            ): item is typeof item & {
              renderer: NonNullable<typeof item.renderer>;
            } => item.renderer !== null,
          );

        if (items.length === 0) return null;

        return {
          nodeID: node.id,
          metadata: {
            name: node.data.hardcodedValues?.name || "Output",
            description:
              node.data.hardcodedValues?.description ||
              "Output from the workflow",
          },
          items,
        };
      })
      .filter((group): group is NonNullable<typeof group> => group !== null);
  }, [nodes]);

  const actionItems = useMemo(() => {
    return outputs.flatMap((group) =>
      group.items.map((item) => ({
        value: item.value,
        metadata: group.metadata,
        renderer: item.renderer,
      })),
    );
  }, [outputs]);

  return (
    <Sheet>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <SheetTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                data-id="agent-outputs-button"
                disabled={!flowID || !hasOutputs()}
              >
                <Icon icon={BookOpen01Icon} className="size-4" />
              </Button>
            </SheetTrigger>
          </TooltipTrigger>
          <TooltipContent>
            <p>Workflow Outputs</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <SheetContent className="flex h-full w-full flex-col overflow-hidden sm:max-w-[600px]">
        <SheetHeader className="px-2 py-2">
          <div className="flex items-center justify-between">
            <div>
              <SheetTitle>
                <Text
                  variant="lead-semibold"
                  as="span"
                  tone="inherit"
                  className="block"
                >
                  Run Outputs
                </Text>
              </SheetTitle>
              <SheetDescription className="mt-1 text-muted-foreground">
                <span className="inline-flex items-center gap-1.5">
                  <Text
                    variant="small-medium-tight"
                    as="span"
                    // TODO(tokens): no exact role: bg-yellow-100 #FCEBB0, text-yellow-800 #876A00; dark-mode classes kept as written (QUALITY 43): no exact role: dark:bg-yellow-900/30 #675100, dark:text-yellow-400 #F7CD33
                    className="rounded-md bg-yellow-100 px-2 py-0.5 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400"
                  >
                    Beta
                  </Text>
                  <Text variant="body-tight" as="span" tone="inherit">
                    This feature is in beta and may contain bugs
                  </Text>
                </span>
              </SheetDescription>
            </div>
            {outputs.length > 0 && <OutputActions items={actionItems} />}
          </div>
        </SheetHeader>
        <div className="flex-grow overflow-y-auto px-2 py-2">
          <ScrollArea className="h-full overflow-auto pr-4">
            <div className="space-y-6">
              {outputs && outputs.length > 0 ? (
                outputs.map((group) => (
                  <div key={group.nodeID} className="space-y-2">
                    <div>
                      <Label>
                        <Text
                          variant="large-semibold-tight"
                          as="span"
                          tone="inherit"
                        >
                          {group.metadata.name || "Unnamed Output"}
                        </Text>
                      </Label>
                      {group.metadata.description && (
                        <Label
                          // TODO(tokens): no exact role: text-gray-600 #4B5563
                          className="mt-1 block text-gray-600"
                        >
                          <Text
                            variant="body-medium-tight"
                            as="span"
                            tone="inherit"
                          >
                            {group.metadata.description}
                          </Text>
                        </Label>
                      )}
                    </div>

                    {group.items.map((item) => (
                      <OutputItem
                        key={item.nodeExecID}
                        value={item.value}
                        metadata={group.metadata}
                        renderer={item.renderer}
                      />
                    ))}
                  </div>
                ))
              ) : (
                <div
                  // TODO(tokens): no exact role: text-gray-500 #6B7280
                  className="flex h-full items-center justify-center text-gray-500"
                >
                  <p>No output blocks available.</p>
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
      </SheetContent>
    </Sheet>
  );
};

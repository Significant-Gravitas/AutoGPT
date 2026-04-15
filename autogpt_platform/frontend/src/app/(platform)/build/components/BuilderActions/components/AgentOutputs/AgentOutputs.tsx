import { BlockUIType } from "@/app/(platform)/build/components/types";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Label } from "@/components/__legacy__/ui/label";
import { ScrollArea } from "@/components/__legacy__/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/__legacy__/ui/sheet";
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
import { BookOpenIcon } from "@phosphor-icons/react";
import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";

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
              node.data.hardcodedValues?.description || "Output from the agent",
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
                <BookOpenIcon className="size-4" />
              </Button>
            </SheetTrigger>
          </TooltipTrigger>
          <TooltipContent>
            <p>Agent Outputs</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <SheetContent className="flex h-full w-full flex-col overflow-hidden sm:max-w-[600px]">
        <SheetHeader className="px-2 py-2">
          <div className="flex items-center justify-between">
            <div>
              <SheetTitle className="text-xl">Run Outputs</SheetTitle>
              <SheetDescription className="mt-1 text-sm text-muted-foreground">
                <span className="inline-flex items-center gap-1.5">
                  <span className="rounded-md bg-yellow-100 px-2 py-0.5 text-xs font-medium text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400">
                    Beta
                  </span>
                  <span>This feature is in beta and may contain bugs</span>
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
                      <Label className="text-base font-semibold">
                        {group.metadata.name || "Unnamed Output"}
                      </Label>
                      {group.metadata.description && (
                        <Label className="mt-1 block text-sm text-gray-600">
                          {group.metadata.description}
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
                <div className="flex h-full items-center justify-center text-gray-500">
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

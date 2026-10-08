import { BlockUIType } from "@/app/(platform)/build/components/types";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { ScrollArea } from "@/components/atoms/ScrollArea/ScrollArea";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
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
import { Sheet } from "@/components/molecules/Sheet/Sheet";
import { useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export const AgentOutputs = ({ flowID }: { flowID: string | null }) => {
  const hasOutputs = useGraphStore(useShallow((state) => state.hasOutputs));
  const [open, setOpen] = useState(false);
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
    <>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                variant="outline"
                size="icon-lg"
                data-id="agent-outputs-button"
                aria-label="Agent Outputs"
                disabled={!flowID || !hasOutputs()}
                onClick={() => setOpen(true)}
              >
                <Icon icon={BookOpen01Icon} className="size-4" />
              </Button>
            }
          />
          <TooltipContent>
            <p>Agent Outputs</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <Sheet
        open={open}
        onOpenChange={setOpen}
        title="Run Outputs"
        description={
          <span className="inline-flex items-center gap-1.5">
            <Badge variant="warning" size="small">
              Beta
            </Badge>
            <span>This feature is in beta and may contain bugs</span>
          </span>
        }
        actions={
          outputs.length > 0 ? <OutputActions items={actionItems} /> : null
        }
        className="w-full overflow-hidden sm:max-w-[600px]"
        bodyClassName="px-2 pb-2"
      >
        <div className="grow overflow-y-auto px-2 py-2">
          <ScrollArea className="h-full overflow-auto pr-4">
            <div className="space-y-6">
              {outputs && outputs.length > 0 ? (
                outputs.map((group) => (
                  <div key={group.nodeID} className="space-y-2">
                    <div>
                      <Text variant="large-semibold" as="h3">
                        {group.metadata.name || "Unnamed Output"}
                      </Text>
                      {group.metadata.description && (
                        <Text variant="body" tone="secondary" className="mt-1">
                          {group.metadata.description}
                        </Text>
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
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  <p>No output blocks available.</p>
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
      </Sheet>
    </>
  );
};

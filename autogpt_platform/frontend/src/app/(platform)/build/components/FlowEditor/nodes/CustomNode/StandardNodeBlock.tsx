import { beautifyString, cn } from "@/lib/utils";
import { CustomNodeData } from "./CustomNode";
import { Text } from "@/components/atoms/Text/Text";
import { FormCreator } from "../FormCreator";
import { preprocessInputSchema } from "../../processors/input-schema-pre-processor";
import { Switch } from "@/components/atoms/Switch/Switch";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { OutputHandler } from "../OutputHandler";
import { NodeCost } from "./components/NodeCost";
import { NodeBadges } from "./components/NodeBadges";

type StandardNodeBlockType = {
  data: CustomNodeData;
  selected: boolean;
  nodeId: string;
};
export const StandardNodeBlock = ({
  data,
  selected,
  nodeId,
}: StandardNodeBlockType) => {
  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[nodeId] || false,
  );
  const setShowAdvanced = useNodeStore((state) => state.setShowAdvanced);

  return (
    <div
      className={cn(
        "z-12 rounded-xl bg-gradient-to-br from-white to-slate-50/30 shadow-lg shadow-slate-900/5 ring-1 ring-slate-200/60 backdrop-blur-sm",
        selected && "shadow-2xl ring-2 ring-slate-200",
      )}
    >
      {/* Header */}
      <div className="flex h-auto flex-col gap-2 rounded-xl border-b border-slate-200/50 bg-gradient-to-r from-slate-50/80 to-white/90 px-4 py-4">
        {/* Upper section  */}
        <div className="flex items-center gap-2">
          <Text
            variant="large-semibold"
            className="tracking-tight text-slate-800"
          >
            {beautifyString(data.title)}
          </Text>
          <Text variant="small" className="!font-medium !text-slate-500">
            #{nodeId.split("-")[0]}
          </Text>
        </div>
        {/* Lower section */}
        <div className="flex space-x-2">
          <NodeCost blockCosts={data.costs} nodeId={nodeId} />
          <NodeBadges categories={data.categories} />
        </div>
      </div>

      {/* Input Handles */}
      <div className="bg-white/40 pb-6 pr-6">
        <FormCreator
          jsonSchema={preprocessInputSchema(data.inputSchema)}
          nodeId={nodeId}
          uiType={data.uiType}
        />
      </div>

      {/* Advanced Button */}
      <div className="flex items-center justify-between gap-2 rounded-b-xl border-t border-slate-200/50 bg-gradient-to-r from-slate-50/60 to-white/80 px-5 py-3.5">
        <Text variant="body" className="font-medium text-slate-700">
          Advanced
        </Text>
        <Switch
          onCheckedChange={(checked) => setShowAdvanced(nodeId, checked)}
          checked={showAdvanced}
        />
      </div>

      {/* Output Handles */}
      <OutputHandler outputSchema={data.outputSchema} nodeId={nodeId} />
    </div>
  );
};

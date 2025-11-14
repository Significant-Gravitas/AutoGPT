import { Text } from "@/components/atoms/Text/Text";
import { beautifyString } from "@/lib/utils";
import { NodeCost } from "./NodeCost";
import { NodeBadges } from "./NodeBadges";
import { NodeContextMenu } from "./NodeContextMenu";
import { CustomNodeData } from "../CustomNode";

export const NodeHeader = ({
  data,
  nodeId,
}: {
  data: CustomNodeData;
  nodeId: string;
}) => {
  return (
    <div className="flex h-auto items-start justify-between gap-2 rounded-xlarge border-b border-slate-200/50 bg-gradient-to-r from-slate-50/80 to-white/90 px-4 py-4">
      <div className="flex flex-col gap-2">
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
      <NodeContextMenu
        subGraphID={data.hardcodedValues?.graph_id}
        nodeId={nodeId}
      />
    </div>
  );
};

import { beautifyString, cn } from "@/lib/utils";
import { CustomNodeData } from "./CustomNode";
import { Text } from "@/components/atoms/Text/Text";
import { FormCreator } from "../FormCreator";
import { preprocessInputSchema } from "@/components/renderers/input-renderer/utils/input-schema-pre-processor";
import { Switch } from "@/components/atoms/Switch/Switch";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { OutputHandler } from "../OutputHandler";
import { NodeCost } from "./components/NodeCost";
import { NodeBadges } from "./components/NodeBadges";
import { NodeExecutionBadge } from "./components/NodeExecutionBadge";
import { nodeStyleBasedOnStatus } from "./helpers";
import { NodeDataRenderer } from "./components/NodeOutput/NodeOutput";
import { NodeContextMenu } from "./components/NodeContextMenu";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { isValidUUID } from "@/app/(platform)/chat/helpers";
import Link from "next/link";
import { parseAsString, useQueryStates } from "nuqs";
import { useGetV2GetLibraryAgentByGraphId } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

type WebhookBlockType = {
  data: CustomNodeData;
  selected: boolean;
  nodeId: string;
};
export const WebhookBlock = ({ data, selected, nodeId }: WebhookBlockType) => {
  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[nodeId] || false,
  );
  const setShowAdvanced = useNodeStore((state) => state.setShowAdvanced);
  const status = useNodeStore((state) => state.getNodeStatus(nodeId));
  const isNodeSaved = isValidUUID(nodeId);

  const [{ flowID }] = useQueryStates({
    flowID: parseAsString,
  });

  // for a single agentId, we are fetching everything - need to make it better in the future
  const { data: libraryAgent } = useGetV2GetLibraryAgentByGraphId(
    flowID ?? "",
    {},
    {
      query: {
        select: (x) => {
          return x.data as LibraryAgent;
        },
        enabled: !!flowID,
      },
    },
  );

  return (
    <div
      className={cn(
        "z-12 max-w-[370px] rounded-xlarge shadow-lg shadow-slate-900/5 ring-1 ring-slate-200/60 backdrop-blur-sm",
        selected && "shadow-2xl ring-2 ring-slate-200",
        status && nodeStyleBasedOnStatus[status],
      )}
    >
      <div className="rounded-xlarge bg-white">
        {/* Header */}
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

        <div className="px-4 pt-4">
          <Alert className="mb-3 rounded-xlarge">
            <AlertDescription>
              <Text variant="small-medium">
                You can set up and manage this trigger in your{" "}
                <Link
                  href={
                    libraryAgent
                      ? `/library/agents/${libraryAgent.id}`
                      : "/library"
                  }
                  className="underline"
                >
                  Agent Library
                </Link>
                {!isNodeSaved && " (after saving the graph)"}.
              </Text>
            </AlertDescription>
          </Alert>
        </div>

        <Text variant="small" className="mb-4 ml-6 !text-purple-700">
          Below inputs are only for display purposes and cannot be edited.
        </Text>

        {/* Input Handles */}
        <div className="pointer-events-none bg-white pr-6 opacity-50">
          <FormCreator
            jsonSchema={preprocessInputSchema(data.inputSchema)}
            nodeId={nodeId}
            uiType={data.uiType}
            showHandles={false}
          />
        </div>

        {/* Advanced Button */}
        <div className="flex items-center justify-between gap-2 border-t border-slate-200/50 bg-white px-5 py-3.5">
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

        <NodeDataRenderer nodeId={nodeId} />
      </div>
      {status && <NodeExecutionBadge status={status} />}
    </div>
  );
};

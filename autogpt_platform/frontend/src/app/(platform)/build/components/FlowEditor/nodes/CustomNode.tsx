import React from "react";
import { Node as XYNode, NodeProps } from "@xyflow/react";
import { FormCreator } from "./FormCreator";
import { RJSFSchema } from "@rjsf/utils";
import { Text } from "@/components/atoms/Text/Text";

import { Switch } from "@/components/atoms/Switch/Switch";
import { preprocessInputSchema } from "../processors/input-schema-pre-processor";
import { OutputHandler } from "./OutputHandler";
import { useNodeStore } from "../../../stores/nodeStore";

export type CustomNodeData = {
  hardcodedValues: {
    [key: string]: any;
  };
  title: string;
  description: string;
  inputSchema: RJSFSchema;
  outputSchema: RJSFSchema;
};

export type CustomNode = XYNode<CustomNodeData, "custom">;

export const CustomNode: React.FC<NodeProps<CustomNode>> = React.memo(
  ({ data, id }) => {
    const showAdvanced = useNodeStore(
      (state) => state.nodeAdvancedStates[id] || false,
    );
    const setShowAdvanced = useNodeStore((state) => state.setShowAdvanced);

    return (
      <div className="rounded-xl border border-slate-200/60 bg-gradient-to-br from-white to-slate-50/30 shadow-lg shadow-slate-900/5 backdrop-blur-sm">
        {/* Header */}
        <div className="flex h-14 items-center justify-center rounded-xl border-b border-slate-200/50 bg-gradient-to-r from-slate-50/80 to-white/90">
          <Text
            variant="large-semibold"
            className="tracking-tight text-slate-800"
          >
            {data.title} #{id}
          </Text>
        </div>

        {/* Input Handles */}
        <div className="bg-white/40 pb-6 pr-6">
          <FormCreator
            jsonSchema={preprocessInputSchema(data.inputSchema)}
            nodeId={id}
          />
        </div>

        {/* Advanced Button */}
        <div className="flex items-center justify-between gap-2 rounded-b-xl border-t border-slate-200/50 bg-gradient-to-r from-slate-50/60 to-white/80 px-5 py-3.5">
          <Text variant="body" className="font-medium text-slate-700">
            Advanced
          </Text>
          <Switch
            onCheckedChange={(checked) => setShowAdvanced(id, checked)}
            checked={showAdvanced}
          />
        </div>

        {/* Output Handles */}
        <OutputHandler outputSchema={data.outputSchema} nodeId={id} />
      </div>
    );
  },
);

CustomNode.displayName = "CustomNode";

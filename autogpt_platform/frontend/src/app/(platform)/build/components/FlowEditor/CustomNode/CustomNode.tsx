import { Node as XYNode, NodeProps } from "@xyflow/react";
import { FormCreator } from "./FormCreator";
import { RJSFSchema } from "@rjsf/utils";

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

export const CustomNode: React.FC<NodeProps<CustomNode>> = ({ data }) => {
  return (
    <div className="min-h-[120px] min-w-[200px] rounded-lg border-2 border-gray-300 bg-white p-6 shadow-lg">
      <FormCreator jsonSchema={data.inputSchema} />
    </div>
  );
};

import { Node as XYNode, NodeProps } from "@xyflow/react";

export type CustomNodeData = {
  hardcodedValues: {
    [key: string]: any;
  };
  title: string;
  description: string;
  inputSchema: Record<string, any>;
  outputSchema: Record<string, any>;
};

export type CustomNode = XYNode<CustomNodeData, "custom">;

export const CustomNode: React.FC<NodeProps<CustomNode>> = ({ data }) => {
  return (
    <div className="min-h-[120px] min-w-[200px] rounded-lg border-2 border-gray-300 bg-white p-6 shadow-lg">
      <div className="mb-2 text-lg font-semibold text-black">{data.title}</div>
      <div className="text-sm text-gray-600">{data.description}</div>
    </div>
  );
};

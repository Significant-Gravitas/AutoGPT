import React, { useCallback } from "react";
import { Handle, Position, useReactFlow, Node } from "@xyflow/react";

export interface BuildCustomNodeData extends Record<string, unknown> {
  label: string; // Label to be displayed in the node
}

const BuildCustomNode: React.FC<Node<BuildCustomNodeData>> = ({ id, data }) => {
  const { fitView } = useReactFlow();

  const handleClick = useCallback(() => {
    console.log(`Fitting view on Node: ${id}`);
    fitView({ duration: 800 });
  }, [id, fitView]);

  return (
    <div
      style={{
        border: "1px solid #ddd",
        padding: "10px",
        borderRadius: "4px",
        background: "white",
      }}
    >
      <Handle type="target" position={Position.Top} />
      <div>{data.label}</div>
      <button onClick={handleClick} style={{ marginTop: "5px" }}>
        Focus Node
      </button>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};

export default BuildCustomNode;

import { CircleIcon } from "@phosphor-icons/react";
import { Handle, Position } from "@xyflow/react";
import { generateHandleId, HandleIdType } from "./helpers";
import { useEdgeStore } from "../../../stores/edgeStore";

const InputNodeHandle = ({
  fieldPathId,
  nodeId,
}: {
  fieldPathId: string;
  nodeId: string;
}) => {
  const extracted_id = fieldPathId
    .split("__")
    .slice(0, -1)
    .join("__")
    .split("_")
    .slice(1)
    .join("_"); // here i am removing root prefix from the id
  const isInputConnected = useEdgeStore((state) =>
    state.isInputConnected(nodeId, extracted_id),
  );
  console.log("extracted_id", extracted_id);
  return (
    <Handle
      type={"target"}
      position={Position.Left}
      id={extracted_id}
      className={"-ml-6 mr-2"}
    >
      <div className="pointer-events-none">
        <CircleIcon
          size={16}
          weight={isInputConnected ? "fill" : "duotone"}
          className={"text-gray-400 opacity-100"}
        />
      </div>
    </Handle>
  );
};

const OutputNodeHandle = ({
  field_name,
  nodeId,
}: {
  field_name: string;
  nodeId: string;
}) => {
  const isOutputConnected = useEdgeStore((state) =>
    state.isInputConnected(nodeId, field_name),
  );
  return (
    <Handle
      type={"source"}
      position={Position.Right}
      id={field_name}
      className={"-mr-2 ml-2"}
    >
      <div className="pointer-events-none">
        <CircleIcon
          size={16}
          weight={isOutputConnected ? "fill" : "duotone"}
          className={"text-gray-400 opacity-100"}
        />
      </div>
    </Handle>
  );
};

export { InputNodeHandle, OutputNodeHandle };

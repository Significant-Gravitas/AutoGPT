import { CircleIcon } from "@phosphor-icons/react";
import { Handle, Position } from "@xyflow/react";
import { useEdgeStore } from "../../../stores/edgeStore";
import { cleanUpHandleId } from "@/components/renderers/InputRenderer/helpers";
import { cn } from "@/lib/utils";

const InputNodeHandle = ({
  handleId,
  nodeId,
}: {
  handleId: string;
  nodeId: string;
}) => {
  const cleanedHandleId = cleanUpHandleId(handleId);
  const isInputConnected = useEdgeStore((state) =>
    state.isInputConnected(nodeId ?? "", cleanedHandleId),
  );

  return (
    <Handle
      type={"target"}
      position={Position.Left}
      id={cleanedHandleId}
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
  hexColor,
}: {
  field_name: string;
  nodeId: string;
  hexColor: string;
}) => {
  const isOutputConnected = useEdgeStore((state) =>
    state.isOutputConnected(nodeId, field_name),
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
          weight={"duotone"}
          color={isOutputConnected ? hexColor : "gray"}
          className={cn("text-gray-400 opacity-100")}
        />
      </div>
    </Handle>
  );
};

export { InputNodeHandle, OutputNodeHandle };

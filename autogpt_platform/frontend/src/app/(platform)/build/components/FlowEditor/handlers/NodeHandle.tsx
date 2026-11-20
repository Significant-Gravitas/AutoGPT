import { Handle, Position } from "@xyflow/react";
import { useEdgeStore } from "../../../stores/edgeStore";
import { cleanUpHandleId } from "@/components/renderers/InputRenderer/helpers";
import { useNodeStore } from "../../../stores/nodeStore";
import { CircleIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
  const isInputBroken = useNodeStore((state) =>
    state.isInputBroken(nodeId, cleanedHandleId),
  );

  return (
    <Handle
      type={"target"}
      position={Position.Left}
      id={cleanedHandleId}
      className={"mr-2 -ml-6"}
      data-tutorial-id={`input-handler-${nodeId}-${cleanedHandleId}`}
    >
      <div className="pointer-events-none">
        <Icon
          icon={CircleIcon}
          size={16}
          fill={isInputConnected ? "currentColor" : "none"}
          tone={isInputBroken ? "error" : undefined}
          className={isInputBroken ? undefined : "text-handle"}
        />
      </div>
    </Handle>
  );
};

const OutputNodeHandle = ({
  field_name,
  nodeId,
  isBroken,
}: {
  field_name: string;
  nodeId: string;
  isBroken: boolean;
}) => {
  return (
    <Handle
      type={"source"}
      position={Position.Right}
      id={field_name}
      className={"-mr-2 ml-2"}
      data-tutorial-id={`output-handler-${nodeId}-${field_name}`}
    >
      <div className="pointer-events-none">
        <Icon
          icon={CircleIcon}
          size={16}
          tone={isBroken ? "error" : undefined}
          className={isBroken ? undefined : "text-handle"}
        />
      </div>
    </Handle>
  );
};

export { InputNodeHandle, OutputNodeHandle };

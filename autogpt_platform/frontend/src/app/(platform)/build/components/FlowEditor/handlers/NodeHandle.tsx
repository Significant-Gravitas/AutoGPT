import { Handle, Position } from "@xyflow/react";
import { useEdgeStore } from "../../../stores/edgeStore";
import { cleanUpHandleId } from "@/components/renderers/InputRenderer/helpers";
import { cn } from "@/lib/utils";
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
      className={"-ml-6 mr-2"}
      data-tutorial-id={`input-handler-${nodeId}-${cleanedHandleId}`}
    >
      <div className="pointer-events-none">
        <Icon
          icon={CircleIcon}
          size={16}
          fill={isInputConnected ? "currentColor" : "none"}
          className={cn(
            "text-handle opacity-100",
            isInputBroken && "text-error",
          )}
        />
      </div>
    </Handle>
  );
};

const OutputNodeHandle = ({
  field_name,
  nodeId,
  hexColor,
  isBroken,
}: {
  field_name: string;
  nodeId: string;
  hexColor: string;
  isBroken: boolean;
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
      data-tutorial-id={`output-handler-${nodeId}-${field_name}`}
    >
      <div className="pointer-events-none">
        <Icon
          icon={CircleIcon}
          size={16}
          // TODO(tokens): no exact role: "gray" #808080 (an unconnected output handle; the input handle is text-handle #9CA3AF)
          color={isOutputConnected ? hexColor : "gray"}
          className={cn("text-handle opacity-100", isBroken && "text-error")}
        />
      </div>
    </Handle>
  );
};

export { InputNodeHandle, OutputNodeHandle };

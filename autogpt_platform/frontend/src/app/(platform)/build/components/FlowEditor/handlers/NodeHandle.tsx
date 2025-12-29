import { CircleIcon } from "@phosphor-icons/react";
import { Handle, Position } from "@xyflow/react";
import { useEdgeStore } from "../../../stores/edgeStore";
import {
  useArrayItemHandleId,
  useIsArrayItem,
} from "@/components/renderers/input-renderer-2/array/context/array-item-context";
import { generateHandleIdFromTitleId } from "./helpers";

const InputNodeHandle = ({
  titleId,
  additional,
  nodeId,
}: {
  titleId: string;
  additional: boolean;
  nodeId: string;
}) => {
  let handleId = "";
  const arrayItemHandleId = useArrayItemHandleId();
  if (arrayItemHandleId) {
    handleId = arrayItemHandleId;
  } else {
    handleId = generateHandleIdFromTitleId(titleId, {
      isAdditionalProperty: additional,
    });
  }

  const isInputConnected = useEdgeStore((state) =>
    state.isInputConnected(nodeId ?? "", handleId),
  );

  return (
    <Handle
      type={"target"}
      position={Position.Left}
      id={handleId}
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

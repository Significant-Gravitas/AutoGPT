import { CircleIcon } from "@phosphor-icons/react";
import { Handle, Position } from "@xyflow/react";

const NodeHandle = ({
  id,
  isConnected,
  side,
}: {
  id: string;
  isConnected: boolean;
  side: "left" | "right";
}) => {
  return (
    <Handle
      type={side === "left" ? "target" : "source"}
      position={side === "left" ? Position.Left : Position.Right}
      id={id}
      className={side === "left" ? "-ml-4 mr-2" : "-mr-2 ml-2"}
    >
      <CircleIcon
        size={16}
        weight={isConnected ? "fill" : "duotone"}
        className={"text-gray-400 opacity-100"}
      />
    </Handle>
  );
};

export default NodeHandle;

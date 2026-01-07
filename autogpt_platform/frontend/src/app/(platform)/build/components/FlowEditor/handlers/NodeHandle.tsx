import { CircleIcon } from "@phosphor-icons/react";
import { Handle, Position } from "@xyflow/react";
import { cn } from "@/lib/utils";

type NodeHandleProps = {
  handleId: string;
  isConnected: boolean;
  side: "left" | "right";
  isBroken?: boolean;
};

const NodeHandle = ({
  handleId,
  isConnected,
  side,
  isBroken = false,
}: NodeHandleProps) => {
  return (
    <Handle
      type={side === "left" ? "target" : "source"}
      position={side === "left" ? Position.Left : Position.Right}
      id={handleId}
      className={cn(
        side === "left" ? "-ml-4 mr-2" : "-mr-2 ml-2",
        isBroken && "pointer-events-none",
      )}
      isConnectable={!isBroken}
    >
      <div className="pointer-events-none">
        <CircleIcon
          size={16}
          weight={isConnected ? "fill" : "duotone"}
          className={cn(
            "opacity-100",
            isBroken ? "text-red-500" : "text-gray-400",
          )}
        />
      </div>
    </Handle>
  );
};

export default NodeHandle;

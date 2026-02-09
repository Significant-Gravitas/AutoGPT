import { BlockIOSubSchema } from "@/lib/autogpt-server-api/types";
import {
  cn,
  beautifyString,
  getTypeBgColor,
  getTypeTextColor,
  getEffectiveType,
} from "@/lib/utils";
import { FC, memo, useCallback } from "react";
import { Handle, Position } from "@xyflow/react";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";

type HandleProps = {
  keyName: string;
  schema: BlockIOSubSchema;
  isConnected: boolean;
  isRequired?: boolean;
  side: "left" | "right";
  title?: string;
  className?: string;
  isBroken?: boolean;
};

// Move the constant out of the component to avoid re-creation on every render.
const TYPE_NAME: Record<string, string> = {
  string: "text",
  number: "number",
  integer: "integer",
  boolean: "true/false",
  object: "object",
  array: "list",
  null: "null",
};

// Extract and memoize the Dot component so that it doesn't re-render unnecessarily.
const Dot: FC<{ isConnected: boolean; type?: string; isBroken?: boolean }> =
  memo(({ isConnected, type, isBroken }) => {
    const color = isBroken
      ? "border-red-500 bg-red-100 dark:bg-red-900/30"
      : isConnected
        ? getTypeBgColor(type || "any")
        : "border-gray-300 dark:border-gray-600";
    return (
      <div
        className={cn(
          "m-1 h-4 w-4 rounded-full border-2 bg-white transition-colors duration-100 group-hover:bg-gray-300 dark:bg-slate-800 dark:group-hover:bg-gray-700",
          color,
          isBroken && "opacity-50",
        )}
      />
    );
  });
Dot.displayName = "Dot";

const NodeHandle: FC<HandleProps> = ({
  keyName,
  schema,
  isConnected,
  isRequired,
  side,
  title,
  className,
  isBroken = false,
}) => {
  // Extract effective type from schema (handles anyOf/oneOf/allOf wrappers)
  const effectiveType = getEffectiveType(schema);

  const typeClass = `text-sm ${getTypeTextColor(effectiveType || "any")} ${
    side === "left" ? "text-left" : "text-right"
  }`;

  const label = (
    <div className={cn("flex flex-grow flex-row", isBroken && "opacity-50")}>
      <span
        className={cn(
          "data-sentry-unmask text-m green flex items-end pr-2 text-gray-900 dark:text-gray-100",
          className,
          isBroken && "text-red-500 line-through",
        )}
      >
        {title || schema.title || beautifyString(keyName.toLowerCase())}
        {isRequired ? "*" : ""}
      </span>
      <span
        className={cn(
          `${typeClass} data-sentry-unmask flex items-end`,
          isBroken && "text-red-400",
        )}
      >
        ({TYPE_NAME[effectiveType as keyof typeof TYPE_NAME] || "any"})
      </span>
    </div>
  );

  // Use a native HTML onContextMenu handler instead of wrapping a large node with a Radix ContextMenu trigger.
  const handleContextMenu = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      // Optionally, you can trigger a custom, lightweight context menu here.
    },
    [],
  );

  if (side === "left") {
    return (
      <div
        key={keyName}
        className={cn("handle-container", isBroken && "pointer-events-none")}
        onContextMenu={handleContextMenu}
      >
        <Handle
          type="target"
          data-testid={`input-handle-${keyName}`}
          position={Position.Left}
          id={keyName}
          className={cn("group -ml-[38px]", isBroken && "cursor-not-allowed")}
          isConnectable={!isBroken}
        >
          <div className="pointer-events-none flex items-center">
            <Dot
              isConnected={isConnected}
              type={effectiveType}
              isBroken={isBroken}
            />
            {label}
          </div>
        </Handle>
        <InformationTooltip description={schema.description} />
      </div>
    );
  } else {
    return (
      <div
        key={keyName}
        className={cn(
          "handle-container justify-end",
          isBroken && "pointer-events-none",
        )}
        onContextMenu={handleContextMenu}
      >
        <Handle
          type="source"
          data-testid={`output-handle-${keyName}`}
          position={Position.Right}
          id={keyName}
          className={cn("group -mr-[38px]", isBroken && "cursor-not-allowed")}
          isConnectable={!isBroken}
        >
          <div className="pointer-events-none flex items-center">
            {label}
            <Dot
              isConnected={isConnected}
              type={effectiveType}
              isBroken={isBroken}
            />
          </div>
        </Handle>
      </div>
    );
  }
};

export default memo(NodeHandle);

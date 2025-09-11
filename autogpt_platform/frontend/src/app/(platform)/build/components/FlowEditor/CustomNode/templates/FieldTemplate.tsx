import React from "react";
import { FieldTemplateProps } from "@rjsf/utils";
import { InfoIcon } from "@phosphor-icons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Text } from "@/components/atoms/Text/Text";
import { useCustomNodeStore } from "../store/customNodeStore";
import { Handle, Position } from "@xyflow/react";

const FieldTemplate: React.FC<FieldTemplateProps> = ({
  id,
  label,
  required,
  description,
  children,
  schema,
  formContext,
}) => {
  const { getShowAdvanced } = useCustomNodeStore();
  const { nodeId } = formContext;
  if (!getShowAdvanced(nodeId) && schema.advanced === true) {
    return null;
  }
  const Dot: React.FC<{ isConnected?: boolean; type?: string }> = ({
    isConnected = false,
    type = "string",
  }) => (
    <div
      className={`h-3 w-3 rounded-full border-2 ${
        isConnected
          ? "border-gray-400 bg-gray-400"
          : "border-gray-300 bg-white hover:border-gray-400"
      }`}
    />
  );

  return (
    <div className="mt-4 min-w-[300px] max-w-md space-y-1">
      {label && (
        <label htmlFor={id} className="flex items-center gap-1">
          <Handle
            type="target"
            position={Position.Left}
            id={id}
            className="-ml-3.5 mr-2"
          >
            <Dot isConnected={false} type={schema.type as string} />
          </Handle>
          <Text variant="body" className="line-clamp-1">
            {label}
          </Text>
          <Text variant="small" className="!text-green-500">
            ({schema.type})
          </Text>
          {required && <span style={{ color: "red" }}>*</span>}
          {description && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    style={{ marginLeft: 6, cursor: "pointer" }}
                    aria-label="info"
                    tabIndex={0}
                  >
                    <InfoIcon />
                  </span>
                </TooltipTrigger>
                <TooltipContent>{description}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </label>
      )}
      <div className="pl-2">{children}</div>
    </div>
  );
};

export default FieldTemplate;

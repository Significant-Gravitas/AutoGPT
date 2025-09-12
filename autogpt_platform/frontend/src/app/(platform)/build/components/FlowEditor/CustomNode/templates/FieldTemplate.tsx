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
import NodeHandle from "../NodeHandle";

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

  const fieldKey = id?.split("_").slice(1, -1).join("_") || "";
  return (
    <div className="mt-4 w-[400px] space-y-1">
      {label && schema.type && (
        <label htmlFor={fieldKey} className="flex items-center gap-1">
          <NodeHandle id={id} isConnected={false} side="left" />
          <Text variant="body" className="line-clamp-1">
            {label}
          </Text>
          <Text variant="small" className="!text-green-500">
            ({schema.type})
          </Text>
          {required && <span style={{ color: "red" }}>*</span>}
          {description?.props?.description && (
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

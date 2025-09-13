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
import { useHandleStore } from "../../../store/handleStore";
import { useEdgeStore } from "../../../store/edgeStore";

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
  const { isInputConnected } = useEdgeStore();
  const { fromRjsfId } = useHandleStore();
  const { nodeId } = formContext;

  const fieldKey = fromRjsfId(id);
  const isConnected = isInputConnected(nodeId, fieldKey);

  if (!getShowAdvanced(nodeId) && schema.advanced === true) {
    return null;
  }

  return (
    <div className="mt-4 w-[400px] space-y-1">
      {label && schema.type && (
        <label htmlFor={id} className="flex items-center gap-1">
          <NodeHandle id={fieldKey} isConnected={isConnected} side="left" />
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
      {!isConnected && <div className="pl-2">{children}</div>}{" "}
    </div>
  );
};

export default FieldTemplate;

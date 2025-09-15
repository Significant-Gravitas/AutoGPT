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

import NodeHandle from "../../handlers/NodeHandle";
import { fromRjsfId } from "../../handlers/helpers";
import { useEdgeStore } from "../../../store/edgeStore";
import { useNodeStore } from "../../../store/nodeStore";

const FieldTemplate: React.FC<FieldTemplateProps> = ({
  id,
  label,
  required,
  description,
  children,
  schema,
  formContext,
}) => {
  const { isInputConnected } = useEdgeStore();
  const { nodeId } = formContext;

  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[nodeId] ?? false,
  );

  const fieldKey = fromRjsfId(id);
  const isConnected = isInputConnected(nodeId, fieldKey);

  if (!showAdvanced && schema.advanced === true) {
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

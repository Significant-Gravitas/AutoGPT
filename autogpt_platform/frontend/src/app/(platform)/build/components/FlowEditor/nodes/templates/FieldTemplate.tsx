import React, { useContext } from "react";
import { FieldTemplateProps } from "@rjsf/utils";
import { InfoIcon } from "@phosphor-icons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Text } from "@/components/atoms/Text/Text";

import NodeHandle from "../../handlers/NodeHandle";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { generateHandleId } from "../../handlers/helpers";
import { getTypeDisplayInfo } from "../helpers";
import { ArrayEditorContext } from "../../components/ArrayEditor/ArrayEditorContext";

const FieldTemplate: React.FC<FieldTemplateProps> = ({
  id,
  label,
  required,
  description,
  children,
  schema,
  formContext,
  uiSchema,
}) => {
  const { isInputConnected } = useEdgeStore();
  const { nodeId } = formContext;

  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[nodeId] ?? false,
  );

  const {
    isArrayItem,
    fieldKey: arrayFieldKey,
    isConnected: isArrayItemConnected,
  } = useContext(ArrayEditorContext);

  let fieldKey = generateHandleId(id);
  let isConnected = isInputConnected(nodeId, fieldKey);
  if (isArrayItem) {
    fieldKey = arrayFieldKey;
    isConnected = isArrayItemConnected;
  }
  const isAnyOf = Array.isArray((schema as any)?.anyOf);
  const isOneOf = Array.isArray((schema as any)?.oneOf);
  const suppressHandle = isAnyOf || isOneOf;

  if (!showAdvanced && schema.advanced === true && !isConnected) {
    return null;
  }

  const fromAnyOf =
    Boolean((uiSchema as any)?.["ui:options"]?.fromAnyOf) ||
    Boolean((formContext as any)?.fromAnyOf);

  const { displayType, colorClass } = getTypeDisplayInfo(schema);

  return (
    <div className="mt-4 w-[400px] space-y-1">
      {label && schema.type && (
        <label htmlFor={id} className="flex items-center gap-1">
          {!suppressHandle && !fromAnyOf && (
            <NodeHandle id={fieldKey} isConnected={isConnected} side="left" />
          )}
          {!fromAnyOf && (
            <Text variant="body" className="line-clamp-1">
              {label}
            </Text>
          )}
          {!fromAnyOf && (
            <Text variant="small" className={colorClass}>
              ({displayType})
            </Text>
          )}
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
      {(isAnyOf || !isConnected) && <div className="pl-2">{children}</div>}{" "}
    </div>
  );
};

export default FieldTemplate;

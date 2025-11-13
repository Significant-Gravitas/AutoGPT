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

import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { generateHandleId } from "@/app/(platform)/build/components/FlowEditor/handlers/helpers";
import { getTypeDisplayInfo } from "@/app/(platform)/build/components/FlowEditor/nodes/helpers";
import { ArrayEditorContext } from "../widgets/ArrayEditorWidget/ArrayEditorContext";
import {
  isCredentialFieldSchema,
  toDisplayName,
  getCredentialProviderFromSchema,
} from "../fields/CredentialField/helpers";
import { cn } from "@/lib/utils";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { BlockUIType } from "@/lib/autogpt-server-api";
import NodeHandle from "@/app/(platform)/build/components/FlowEditor/handlers/NodeHandle";

const FieldTemplate: React.FC<FieldTemplateProps> = ({
  id: fieldId,
  label,
  required,
  description,
  children,
  schema,
  formContext,
  uiSchema,
}) => {
  const { isInputConnected } = useEdgeStore();
  const { nodeId, showHandles = true, size = "small" } = formContext;

  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[nodeId] ?? false,
  );

  const { isArrayItem, arrayFieldHandleId } = useContext(ArrayEditorContext);

  const isAnyOf = Array.isArray((schema as any)?.anyOf);
  const isOneOf = Array.isArray((schema as any)?.oneOf);
  const isCredential = isCredentialFieldSchema(schema);
  const suppressHandle = isAnyOf || isOneOf;

  let handleId = null;
  if (!isArrayItem) {
    handleId = generateHandleId(fieldId);
  } else {
    handleId = arrayFieldHandleId;
  }

  const isConnected = showHandles ? isInputConnected(nodeId, handleId) : false;

  if (!showAdvanced && schema.advanced === true && !isConnected) {
    return null;
  }

  const fromAnyOf =
    Boolean((uiSchema as any)?.["ui:options"]?.fromAnyOf) ||
    Boolean((formContext as any)?.fromAnyOf);

  const { displayType, colorClass } = getTypeDisplayInfo(schema);

  let credentialProvider = null;
  if (isCredential) {
    credentialProvider = getCredentialProviderFromSchema(
      useNodeStore.getState().getHardCodedValues(nodeId),
      schema as BlockIOCredentialsSubSchema,
    );
  }
  if (formContext.uiType === BlockUIType.NOTE) {
    return <div className="w-full space-y-1">{children}</div>;
  }

  // Size-based styling

  const shouldShowHandle =
    showHandles && !suppressHandle && !fromAnyOf && !isCredential;

  return (
    <div
      className={cn(
        "mb-4 space-y-2",
        fromAnyOf && "mb-0",
        size === "small" ? "w-[350px]" : "w-full",
      )}
    >
      {label && schema.type && (
        <label htmlFor={fieldId} className="flex items-center gap-1">
          {shouldShowHandle && (
            <NodeHandle
              handleId={handleId}
              isConnected={isConnected}
              side="left"
            />
          )}
          {!fromAnyOf && (
            <Text
              variant="body"
              className={cn(
                "line-clamp-1",
                isCredential && !shouldShowHandle && "ml-3",
                size == "large" && "ml-0",
              )}
            >
              {isCredential && credentialProvider
                ? toDisplayName(credentialProvider) + " credentials"
                : label}
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
      {(isAnyOf || !isConnected) && (
        <div className={cn(size === "small" ? "pl-2" : "")}>{children}</div>
      )}{" "}
    </div>
  );
};

export default FieldTemplate;

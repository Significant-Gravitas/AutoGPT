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
import {
  isCredentialFieldSchema,
  toDisplayName,
  getCredentialProviderFromSchema,
} from "../fields/CredentialField/helpers";
import { cn } from "@/lib/utils";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { BlockUIType } from "@/lib/autogpt-server-api";

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
  const { nodeId } = formContext;

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

  const isConnected = isInputConnected(nodeId, handleId);

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

  return (
    <div className="mt-4 w-[350px] space-y-1">
      {label && schema.type && (
        <label htmlFor={fieldId} className="flex items-center gap-1">
          {!suppressHandle && !fromAnyOf && !isCredential && (
            <NodeHandle
              handleId={handleId}
              isConnected={isConnected}
              side="left"
            />
          )}
          {!fromAnyOf && (
            <Text
              variant="body"
              className={cn("line-clamp-1", isCredential && "ml-3")}
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
      {(isAnyOf || !isConnected) && <div className="pl-2">{children}</div>}{" "}
    </div>
  );
};

export default FieldTemplate;

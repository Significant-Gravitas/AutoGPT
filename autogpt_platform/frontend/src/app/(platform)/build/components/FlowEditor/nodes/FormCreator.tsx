import { RJSFSchema } from "@rjsf/utils";
import React, { useCallback, useRef } from "react";
import { uiSchema } from "./uiSchema";
import { useNodeStore } from "../../../stores/nodeStore";
import { BlockUIType } from "../../types";
import { FormRenderer } from "@/components/renderers/InputRenderer/FormRenderer";

interface FormCreatorProps {
  jsonSchema: RJSFSchema;
  nodeId: string;
  uiType: BlockUIType;
  /** When true the block is an MCP Tool with a selected tool. */
  isMCPWithTool?: boolean;
  showHandles?: boolean;
  className?: string;
}

export const FormCreator: React.FC<FormCreatorProps> = React.memo(
  ({
    jsonSchema,
    nodeId,
    uiType,
    isMCPWithTool = false,
    showHandles = true,
    className,
  }) => {
    const updateNodeData = useNodeStore((state) => state.updateNodeData);

    const getHardCodedValues = useNodeStore(
      (state) => state.getHardCodedValues,
    );

    const isAgent = uiType === BlockUIType.AGENT;

    // Use useCallback to stabilize the handleChange function reference
    // This prevents unnecessary re-renders of the Form component
    const handleChange = useCallback(
      ({ formData }: any) => {
        if ("credentials" in formData && !formData.credentials?.id) {
          delete formData.credentials;
        }

        let updatedValues;
        if (isAgent) {
          updatedValues = {
            ...getHardCodedValues(nodeId),
            inputs: formData,
          };
        } else if (isMCPWithTool) {
          // Separate credentials from tool arguments — credentials are stored
          // at the top level of hardcodedValues, not inside tool_arguments.
          const { credentials, ...toolArgs } = formData;
          updatedValues = {
            ...getHardCodedValues(nodeId),
            tool_arguments: toolArgs,
            ...(credentials?.id ? { credentials } : {}),
          };
        } else {
          updatedValues = formData;
        }

        updateNodeData(nodeId, { hardcodedValues: updatedValues });
      },
      [nodeId, isAgent, isMCPWithTool, getHardCodedValues, updateNodeData],
    );

    const hardcodedValues = getHardCodedValues(nodeId);

    let initialValues;
    if (isAgent) {
      initialValues = hardcodedValues.inputs ?? {};
    } else if (isMCPWithTool) {
      // Merge tool arguments with credentials for the form
      initialValues = {
        ...(hardcodedValues.tool_arguments ?? {}),
        ...(hardcodedValues.credentials?.id
          ? { credentials: hardcodedValues.credentials }
          : {}),
      };
    } else {
      initialValues = hardcodedValues;
    }

    // Use useRef to maintain stable initialValues reference across renders
    // Only update when the serialized values actually change
    const initialValuesRef = useRef<Record<string, any>>({});
    const initialValuesStr = JSON.stringify(initialValues);
    const currentStr = JSON.stringify(initialValuesRef.current);

    if (initialValuesStr !== currentStr) {
      initialValuesRef.current = initialValues;
    }

    return (
      <div
        className={className}
        data-id={`form-creator-container-${nodeId}-node`}
      >
        <FormRenderer
          jsonSchema={jsonSchema}
          handleChange={handleChange}
          uiSchema={uiSchema}
          initialValues={initialValuesRef.current}
          formContext={{
            nodeId: nodeId,
            uiType: uiType,
            showHandles: showHandles,
            size: "small",
          }}
        />
      </div>
    );
  },
);

FormCreator.displayName = "FormCreator";
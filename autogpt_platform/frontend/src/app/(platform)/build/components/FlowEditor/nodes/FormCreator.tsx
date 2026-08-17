import { RJSFSchema } from "@rjsf/utils";
import React from "react";
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

    const handleChange = ({ formData }: any) => {
      // RJSF seeds `const` provider/type into default form state, so an
      // untouched credential field arrives as {provider, type} with no id.
      // That half object must never reach input_default: graph activation
      // indexes creds_meta["id"] and would raise KeyError. Field names follow
      // the backend rule in data/model.py:is_credentials_field_name.
      for (const key of Object.keys(formData)) {
        const isCredentialField =
          key === "credentials" || key.endsWith("_credentials");
        if (isCredentialField && !formData[key]?.id) {
          delete formData[key];
        }
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
    };

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

    return (
      <div
        className={className}
        data-id={`form-creator-container-${nodeId}-node`}
      >
        <FormRenderer
          jsonSchema={jsonSchema}
          handleChange={handleChange}
          uiSchema={uiSchema}
          initialValues={initialValues}
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

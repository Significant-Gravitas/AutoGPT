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
        // All form fields are tool arguments (credentials handled by dialog)
        updatedValues = {
          ...getHardCodedValues(nodeId),
          tool_arguments: formData,
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
      initialValues = hardcodedValues.tool_arguments ?? {};
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

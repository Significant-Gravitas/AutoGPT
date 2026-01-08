import { RJSFSchema } from "@rjsf/utils";
import React from "react";
import { uiSchema } from "./uiSchema";
import { useNodeStore } from "../../../stores/nodeStore";
import { BlockUIType } from "../../types";
import { FormRenderer } from "@/components/renderers/InputRenderer/FormRenderer";

export const FormCreator = React.memo(
  ({
    jsonSchema,
    nodeId,
    uiType,
    showHandles = true,
    className,
  }: {
    jsonSchema: RJSFSchema;
    nodeId: string;
    uiType: BlockUIType;
    showHandles?: boolean;
    className?: string;
  }) => {
    const updateNodeData = useNodeStore((state) => state.updateNodeData);

    const getHardCodedValues = useNodeStore(
      (state) => state.getHardCodedValues,
    );

    const handleChange = ({ formData }: any) => {
      if ("credentials" in formData && !formData.credentials?.id) {
        delete formData.credentials;
      }

      const updatedValues =
        uiType === BlockUIType.AGENT
          ? {
              ...getHardCodedValues(nodeId),
              inputs: formData,
            }
          : formData;

      updateNodeData(nodeId, { hardcodedValues: updatedValues });
    };

    const hardcodedValues = getHardCodedValues(nodeId);
    const initialValues =
      uiType === BlockUIType.AGENT
        ? (hardcodedValues.inputs ?? {})
        : hardcodedValues;

    return (
      <div className={className}>
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

import { RJSFSchema } from "@rjsf/utils";
import React from "react";
import { uiSchema } from "./uiSchema";
import { useNodeStore } from "../../../stores/nodeStore";
import { BlockUIType } from "../../types";
import { FormRenderer } from "@/components/renderers/input-renderer/FormRenderer";

export const FormCreator = React.memo(
  ({
    jsonSchema,
    nodeId,
    uiType,
  }: {
    jsonSchema: RJSFSchema;
    nodeId: string;
    uiType: BlockUIType;
  }) => {
    const updateNodeData = useNodeStore((state) => state.updateNodeData);
    const getHardCodedValues = useNodeStore(
      (state) => state.getHardCodedValues,
    );
    const handleChange = ({ formData }: any) => {
      if ("credentials" in formData && !formData.credentials?.id) {
        delete formData.credentials;
      }
      updateNodeData(nodeId, { hardcodedValues: formData });
    };

    const initialValues = getHardCodedValues(nodeId);

    return (
      <FormRenderer
        jsonSchema={jsonSchema}
        handleChange={handleChange}
        uiSchema={uiSchema}
        initialValues={initialValues}
        formContext={{
          nodeId: nodeId,
          uiType: uiType,
          showHandles: true,
          size: "small",
        }}
      />
    );
  },
);

FormCreator.displayName = "FormCreator";

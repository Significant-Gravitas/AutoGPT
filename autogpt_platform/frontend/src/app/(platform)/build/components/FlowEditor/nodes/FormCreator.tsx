import Form from "@rjsf/core";
import validator from "@rjsf/validator-ajv8";
import { RJSFSchema } from "@rjsf/utils";
import React from "react";
import { widgets } from "./widgets";
import { fields } from "./fields";
import { templates } from "./templates";
import { uiSchema } from "./uiSchema";
import { useNodeStore } from "../../../stores/nodeStore";
import { BlockUIType } from "../../types";

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
      updateNodeData(nodeId, { hardcodedValues: formData });
    };

    const initialValues = getHardCodedValues(nodeId);

    return (
      <Form
        schema={jsonSchema}
        validator={validator}
        fields={fields}
        templates={templates}
        widgets={widgets}
        formContext={{ nodeId: nodeId, uiType: uiType }}
        onChange={handleChange}
        uiSchema={uiSchema}
        formData={initialValues}
      />
    );
  },
);

FormCreator.displayName = "FormCreator";

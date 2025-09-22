import Form from "@rjsf/core";
import validator from "@rjsf/validator-ajv8";
import { RJSFSchema } from "@rjsf/utils";
import React from "react";
import { widgets } from "./widgets";
import { fields } from "./fields";
import { templates } from "./templates";
import { uiSchema } from "./uiSchema";
import { useNodeStore } from "../../../stores/nodeStore";

export const FormCreator = React.memo(
  ({ jsonSchema, nodeId }: { jsonSchema: RJSFSchema; nodeId: string }) => {
    const updateNodeData = useNodeStore((state) => state.updateNodeData);
    const handleChange = ({ formData }: any) => {
      updateNodeData(nodeId, { hardcodedValues: formData });
    };

    return (
      <Form
        schema={jsonSchema}
        validator={validator}
        fields={fields}
        templates={templates}
        widgets={widgets}
        formContext={{ nodeId: nodeId }}
        onChange={handleChange}
        uiSchema={uiSchema}
      />
    );
  },
);

FormCreator.displayName = "FormCreator";

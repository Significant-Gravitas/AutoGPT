import Form from "@rjsf/core";
import validator from "@rjsf/validator-ajv8";
import { RJSFSchema } from "@rjsf/utils";
import React from "react";
import { widgets } from "../../../../../../components/form-renderer/widgets";
import { fields } from "../../../../../../components/form-renderer/fields";
import { templates } from "../../../../../../components/form-renderer/templates";
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
      if ("credentials" in formData && !formData.credentials?.id) {
        delete formData.credentials;
      }
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
        formContext={{
          nodeId: nodeId,
          uiType: uiType,
          showHandles: true,
          size: "small",
        }}
        onChange={handleChange}
        uiSchema={uiSchema}
        formData={initialValues}
      />
    );
  },
);

FormCreator.displayName = "FormCreator";

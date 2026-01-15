import { RJSFSchema } from "@rjsf/utils";
import { preprocessInputSchema } from "./utils/input-schema-pre-processor";
import { useMemo } from "react";
import { customValidator } from "./utils/custom-validator";
import Form from "./registry";
import { ExtendedFormContextType } from "./types";
import { generateUiSchemaForCustomFields } from "./utils/generate-ui-schema";

type FormRendererProps = {
  jsonSchema: RJSFSchema;
  handleChange: (formData: any) => void;
  uiSchema: any;
  initialValues: any;
  formContext: ExtendedFormContextType;
};

export const FormRenderer = ({
  jsonSchema,
  handleChange,
  uiSchema,
  initialValues,
  formContext,
}: FormRendererProps) => {
  const preprocessedSchema = useMemo(() => {
    return preprocessInputSchema(jsonSchema);
  }, [jsonSchema]);

  // Merge custom field ui:field settings with existing uiSchema
  const mergedUiSchema = useMemo(() => {
    return generateUiSchemaForCustomFields(preprocessedSchema, uiSchema);
  }, [preprocessedSchema, uiSchema]);

  return (
    <div className={"mb-6 mt-4"} data-tutorial-id="input-handles">
      <Form
        formContext={formContext}
        idPrefix="agpt"
        idSeparator="_%_"
        schema={preprocessedSchema}
        validator={customValidator}
        onChange={handleChange}
        uiSchema={mergedUiSchema}
        formData={initialValues}
        liveValidate={false}
      />
    </div>
  );
};

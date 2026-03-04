import { cn } from "@/lib/utils";
import { RJSFSchema } from "@rjsf/utils";
import { useMemo } from "react";
import Form from "./registry";
import { ExtendedFormContextType } from "./types";
import { customValidator } from "./utils/custom-validator";
import { generateUiSchemaForCustomFields } from "./utils/generate-ui-schema";
import { preprocessInputSchema } from "./utils/input-schema-pre-processor";

type FormRendererProps = {
  jsonSchema: RJSFSchema;
  handleChange: (formData: any) => void;
  uiSchema: any;
  initialValues: any;
  formContext: ExtendedFormContextType;
  className?: string;
};

export function FormRenderer({
  jsonSchema,
  handleChange,
  uiSchema,
  initialValues,
  formContext,
  className,
}: FormRendererProps) {
  const preprocessedSchema = useMemo(() => {
    return preprocessInputSchema(jsonSchema);
  }, [jsonSchema]);

  // Merge custom field ui:field settings with existing uiSchema
  const mergedUiSchema = useMemo(() => {
    return generateUiSchemaForCustomFields(preprocessedSchema, uiSchema);
  }, [preprocessedSchema, uiSchema]);

  return (
    <div
      className={cn("mb-6 mt-4", className)}
      data-tutorial-id="input-handles"
    >
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
}

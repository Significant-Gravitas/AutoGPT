import { RJSFSchema } from "@rjsf/utils";
import { preprocessInputSchema } from "./utils/input-schema-pre-processor";
import { useMemo } from "react";
import { customValidator } from "./utils/custom-validator";
import Form from "./registry";
import { ExtendedFormContextType } from "./types";

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

  return (
    <div className={"mb-6 mt-4"}>
      <Form
        formContext={formContext}
        idPrefix="agpt"
        idSeparator="_%_"
        schema={preprocessedSchema}
        validator={customValidator}
        onChange={handleChange}
        uiSchema={uiSchema}
        formData={initialValues}
        liveValidate={false}
      />
    </div>
  );
};

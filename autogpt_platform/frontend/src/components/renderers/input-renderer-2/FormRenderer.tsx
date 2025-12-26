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

export const FormRenderer2 = ({
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
    <div className={"mt-4"}>
      <Form
        formContext={formContext}
        schema={preprocessedSchema}
        validator={customValidator}
        onChange={handleChange}
        uiSchema={uiSchema}
        formData={initialValues}
        noValidate={true}
        liveValidate={false}
      />
    </div>
  );
};

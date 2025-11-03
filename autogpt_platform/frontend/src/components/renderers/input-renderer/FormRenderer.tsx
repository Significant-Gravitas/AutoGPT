import { BlockUIType } from "@/app/(platform)/build/components/types";
import validator from "@rjsf/validator-ajv8";
import Form from "@rjsf/core";
import { RJSFSchema } from "@rjsf/utils";
import { fields } from "./fields";
import { templates } from "./templates";
import { widgets } from "./widgets";
import { preprocessInputSchema } from "./utils/input-schema-pre-processor";
import { useMemo } from "react";

type FormContextType = {
  nodeId?: string;
  uiType?: BlockUIType;
  showHandles?: boolean;
  size?: "small" | "medium" | "large";
};

type FormRendererProps = {
  jsonSchema: RJSFSchema;
  handleChange: (formData: any) => void;
  uiSchema: any;
  initialValues: any;
  formContext: FormContextType;
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
    <div className={"mt-4"}>
      <Form
        schema={preprocessedSchema}
        validator={validator}
        fields={fields}
        templates={templates}
        widgets={widgets}
        formContext={formContext}
        onChange={handleChange}
        uiSchema={uiSchema}
        formData={initialValues}
      />
    </div>
  );
};

import { BlockUIType } from "@/app/(platform)/build/components/types";
import Form from "@rjsf/core";
import { RJSFSchema } from "@rjsf/utils";
import { fields } from "./fields";
import { templates } from "./templates";
import { widgets } from "./widgets";
import { preprocessInputSchema } from "./utils/input-schema-pre-processor";
import { useMemo } from "react";
import { customValidator } from "./utils/custom-validator";

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
        validator={customValidator}
        fields={fields}
        templates={templates}
        widgets={widgets}
        formContext={formContext}
        onChange={handleChange}
        uiSchema={uiSchema}
        formData={initialValues}
        noValidate={true}
        liveValidate={false}
      />
    </div>
  );
};

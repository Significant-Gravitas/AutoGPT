import { RJSFSchema } from "@rjsf/utils";
import { preprocessInputSchema } from "./utils/input-schema-pre-processor";
import { useMemo } from "react";
import { customValidator } from "./utils/custom-validator";
import { isLlmModelFieldSchema } from "./custom/LlmModelField/LlmModelField";
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

  const llmModelUiSchema = useMemo(() => {
    return buildLlmModelUiSchema(preprocessedSchema);
  }, [preprocessedSchema]);

  const mergedUiSchema = useMemo(() => {
    return mergeUiSchema(uiSchema, llmModelUiSchema);
  }, [uiSchema, llmModelUiSchema]);
  return (
    <div className={"mb-6 mt-4"}>
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

function buildLlmModelUiSchema(schema: RJSFSchema): Record<string, any> {
  if (!schema || typeof schema !== "object") {
    return {};
  }

  if (isLlmModelFieldSchema(schema)) {
    return { "ui:field": "custom/llm_model_field" };
  }

  const result: Record<string, any> = {};

  if (schema.type === "object" && schema.properties) {
    for (const [key, property] of Object.entries(schema.properties)) {
      if (property && typeof property === "object") {
        const nestedSchema = buildLlmModelUiSchema(property as RJSFSchema);
        if (Object.keys(nestedSchema).length > 0) {
          result[key] = nestedSchema;
        }
      }
    }
  }

  if (schema.type === "array" && schema.items) {
    const nestedSchema = buildLlmModelUiSchema(schema.items as RJSFSchema);
    if (Object.keys(nestedSchema).length > 0) {
      result.items = nestedSchema;
    }
  }

  return result;
}

function mergeUiSchema(base: any, overrides: any): any {
  if (!overrides || typeof overrides !== "object") {
    return base ?? overrides;
  }

  const result: Record<string, any> = { ...(base || {}) };
  for (const [key, value] of Object.entries(overrides)) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      result[key] = mergeUiSchema((base || {})[key], value);
    } else {
      result[key] = value;
    }
  }
  return result;
}

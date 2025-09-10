import Form from "@rjsf/core";
import validator from "@rjsf/validator-ajv8";
import { RJSFSchema } from "@rjsf/utils";

export const FormCreator = ({ jsonSchema }: { jsonSchema: RJSFSchema }) => {
  return <Form schema={jsonSchema} validator={validator} />;
};

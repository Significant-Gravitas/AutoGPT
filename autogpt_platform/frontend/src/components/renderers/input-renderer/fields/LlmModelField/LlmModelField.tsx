import { FieldProps } from "@rjsf/utils";

export const LlmModelField = (_props: FieldProps) => {
  return null;
};

export function isLlmModelFieldSchema(schema: unknown): boolean {
  return (
    typeof schema === "object" && schema !== null && "llm_model" in schema
  );
}

import { FieldProps } from "@rjsf/utils";
import { BlockIOObjectSubSchema } from "@/lib/autogpt-server-api/types";

type FormData = Record<string, boolean> | null | undefined;

interface UseMultiSelectFieldOptions {
  schema: BlockIOObjectSubSchema;
  formData: FormData;
}

export function useMultiSelectField({
  schema,
  formData,
}: UseMultiSelectFieldOptions) {
  const getOptionSchema = (): Record<string, BlockIOObjectSubSchema> => {
    if (schema.properties) {
      return schema.properties as Record<string, BlockIOObjectSubSchema>;
    }
    if (
      "anyOf" in schema &&
      Array.isArray(schema.anyOf) &&
      schema.anyOf.length > 0 &&
      "properties" in schema.anyOf[0]
    ) {
      return (schema.anyOf[0] as BlockIOObjectSubSchema).properties as Record<
        string,
        BlockIOObjectSubSchema
      >;
    }
    return {};
  };

  const optionSchema = getOptionSchema();
  const options = Object.keys(optionSchema);

  const getSelection = (): string[] => {
    if (!formData || typeof formData !== "object") {
      return [];
    }
    return Object.entries(formData)
      .filter(([_, value]) => value === true)
      .map(([key]) => key);
  };

  const selection = getSelection();

  const createChangeHandler =
    (
      onChange: FieldProps["onChange"],
      fieldPathId: FieldProps["fieldPathId"],
    ) =>
    (values: string[]) => {
      const newValue = Object.fromEntries(
        options.map((opt) => [opt, values.includes(opt)]),
      );
      onChange(newValue, fieldPathId?.path);
    };

  return {
    optionSchema,
    options,
    selection,
    createChangeHandler,
  };
}

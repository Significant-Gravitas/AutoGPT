import {
  FormContextType,
  MultiSchemaFieldTemplateProps,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import { cn } from "../lib/utils";
import { isOptionalType } from "../../utils/schema-utils";

export default function MultiSchemaFieldTemplate<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
  selector,
  optionSchemaField,
  schema,
}: MultiSchemaFieldTemplateProps<T, S, F>) {
  const isOptional = isOptionalType(schema);

  return (
    <div>
      <div className={cn("mb-4")}>{selector}</div>
      {optionSchemaField}
    </div>
  );
}

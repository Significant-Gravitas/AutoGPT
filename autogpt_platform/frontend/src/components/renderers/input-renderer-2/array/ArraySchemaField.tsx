import {
  FieldProps,
  FormContextType,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import { ArrayItemProvider } from "./context/array-item-context";

const ArraySchemaField = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(
  props: FieldProps<T, S, F>,
) => {
  const { index, registry, name, uiSchema } = props;
  const { SchemaField } = registry.fields;

  return (
    <ArrayItemProvider arrayItemHandleId={`${name.slice(0, -2)}_$_${index}`}>
      <SchemaField {...props} />
    </ArrayItemProvider>
  );
};

export default ArraySchemaField;

import {
  FieldProps,
  FormContextType,
  getUiOptions,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import { ArrayItemProvider } from "./context/array-item-context";
import { ARRAY_ITEM_FLAG, getHandleId, updateUiOption } from "../helpers";

const ArraySchemaField = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(
  props: FieldProps<T, S, F>,
) => {
  const { index, registry, name, fieldPathId } = props;
  const { SchemaField } = registry.fields;

  const uiOptions = getUiOptions<T, S, F>(props.uiSchema);
  const handleId = getHandleId({
    uiOptions,
    id: fieldPathId.$id,
    schema: props.schema,
  });
  const updatedUiSchema = updateUiOption(props.uiSchema, {
    handleId: handleId + ARRAY_ITEM_FLAG,
  });

  return (
    <ArrayItemProvider arrayItemHandleId={`${name.slice(0, -2)}_$_${index}`}>
      <SchemaField {...props} uiSchema={updatedUiSchema} />
    </ArrayItemProvider>
  );
};

export default ArraySchemaField;

import { FieldProps, getUiOptions } from "@rjsf/utils";
import { ArrayItemProvider } from "./context/array-item-context";
import { getHandleId, updateUiOption } from "../../helpers";
import { ARRAY_ITEM_FLAG } from "../../constants";

const ArraySchemaField = (props: FieldProps) => {
  const { index, registry, name, fieldPathId } = props;
  const { SchemaField } = registry.fields;

  const uiOptions = getUiOptions(props.uiSchema);

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

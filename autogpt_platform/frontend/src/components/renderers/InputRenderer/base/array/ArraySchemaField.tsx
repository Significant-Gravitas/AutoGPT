import { FieldProps, getUiOptions } from "@rjsf/utils";
import { getHandleId, updateUiOption } from "../../helpers";
import { ARRAY_ITEM_FLAG } from "../../constants";

const ArraySchemaField = (props: FieldProps) => {
  const { index, registry, fieldPathId } = props;
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
    <SchemaField
      {...props}
      uiSchema={updatedUiSchema}
      title={"_item-" + index.toString()}
    />
  );
};

export default ArraySchemaField;

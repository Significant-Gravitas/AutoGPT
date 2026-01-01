import { RegistryFieldsType } from "@rjsf/utils";
import { AnyOfField } from "../anyof/AnyOfField";
import { ArraySchemaField } from "../array";
import { getCustomFieldComponents } from "./custom-field";

export function generateFields(): RegistryFieldsType {
  return {
    AnyOfField,
    ArraySchemaField,

    // My custom fields
    ...getCustomFieldComponents(),
  };
}

export default generateFields();

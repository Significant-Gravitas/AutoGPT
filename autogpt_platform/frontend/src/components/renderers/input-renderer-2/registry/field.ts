import {
  FormContextType,
  RegistryFieldsType,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import { AnyOfField } from "../anyof/AnyOfField";
import { ArraySchemaField } from "../array";
import { CredentialField } from "../credentials/CredentialField";
import { getCustomFieldComponents } from "./custom-field";

export function generateFields<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(): RegistryFieldsType<T, S, F> {
  return {
    AnyOfField,
    ArraySchemaField,

    // My custom fields
    ...getCustomFieldComponents(),
  };
}

export default generateFields();

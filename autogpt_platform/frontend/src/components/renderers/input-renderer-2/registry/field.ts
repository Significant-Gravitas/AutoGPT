import {
  FormContextType,
  RegistryFieldsType,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import { AnyOfField } from "../anyof/AnyOfField";
import { ArraySchemaField } from "../array";

export function generateFields<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(): RegistryFieldsType<T, S, F> {
  return {
    AnyOfField,
    ArraySchemaField,
  };
}

export default generateFields();

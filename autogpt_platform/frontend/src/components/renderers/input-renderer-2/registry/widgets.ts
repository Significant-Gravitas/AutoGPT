import {
  FormContextType,
  RegistryWidgetsType,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import { ExtendedFormContextType } from "../types";
import {
  TextWidget,
  SelectWidget,
  CheckboxWidget,
  FileWidget,
  DateWidget,
  TimeWidget,
  DateTimeWidget,
} from "../inputs";

export function generateWidgets<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = ExtendedFormContextType,
>(): RegistryWidgetsType<T, S, F> {
  return {
    TextWidget,
    SelectWidget,
    CheckboxWidget,
    FileWidget,
    DateWidget,
    TimeWidget,
    DateTimeWidget,
  };
}

export default generateWidgets();

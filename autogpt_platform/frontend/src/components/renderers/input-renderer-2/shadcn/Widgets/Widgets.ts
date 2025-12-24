import {
  FormContextType,
  RegistryWidgetsType,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import TextWidget from "./components/TextWidget/TextWidget";
import { ExtendedFormContextType } from "../types";
import { FileWidget } from "./components/FileWidget/FileWidget";
import { DateWidget } from "./components/DateWidget/DateWidget";
import { TimeWidget } from "./components/TimeWidget/TimeWidget";
import { DateTimeWidget } from "./components/DateTimeWidget/DateTimeWidget";
import { SelectWidget } from "./components/SelectWidget/SelectWidget";
import { CheckboxWidget } from "./components/CheckBoxWidget/CheckBoxWidget";

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

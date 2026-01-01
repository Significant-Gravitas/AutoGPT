import { RegistryWidgetsType } from "@rjsf/utils";
import {
  TextWidget,
  SelectWidget,
  CheckboxWidget,
  FileWidget,
  DateWidget,
  TimeWidget,
  DateTimeWidget,
} from "../inputs";

export function generateWidgets(): RegistryWidgetsType {
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

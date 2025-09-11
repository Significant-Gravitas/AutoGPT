import { RegistryWidgetsType } from "@rjsf/utils";
import { SelectWidget } from "./SelectWidget";
import { TextInputWidget } from "./TextInputWidget";
import { SwitchWidget } from "./SwitchWidget";
import { FileWidget } from "./FileWidget";

export const widgets: RegistryWidgetsType = {
  TextWidget: TextInputWidget,
  SelectWidget: SelectWidget,
  CheckboxWidget: SwitchWidget,
  FileWidget: FileWidget,
};

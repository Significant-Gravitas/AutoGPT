import { RegistryWidgetsType } from "@rjsf/utils";
import { SelectWidget } from "./SelectWidget";
import { TextInputWidget } from "./TextInputWidget";
import { SwitchWidget } from "./SwitchWidget";
import { FileWidget } from "./FileWidget";
import { DateInputWidget } from "./DateInputWidget";
import { TimeInputWidget } from "./TimeInputWidget";
import { DateTimeInputWidget } from "./DateTimeInputWidget";

export const widgets: RegistryWidgetsType = {
  TextWidget: TextInputWidget,
  SelectWidget: SelectWidget,
  CheckboxWidget: SwitchWidget,
  FileWidget: FileWidget,
  DateWidget: DateInputWidget,
  TimeWidget: TimeInputWidget,
  DateTimeWidget: DateTimeInputWidget,
};

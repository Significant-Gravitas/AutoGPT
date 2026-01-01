import {
  RegistryFieldsType,
  RegistryWidgetsType,
  TemplatesType,
} from "@rjsf/utils";
import { AnyOfField } from "./anyof/AnyOfField";
import { AddButton, CopyButton, RemoveButton } from "./standard/buttons";
import {
  ArrayFieldItemTemplate,
  ArraySchemaField,
  ArrayFieldTemplate,
} from "./array";
import {
  ObjectFieldTemplate,
  OptionalDataControlsTemplate,
  WrapIfAdditionalTemplate,
} from "./object";
import {
  DescriptionField,
  FieldErrorTemplate,
  FieldHelpTemplate,
  FieldTemplate,
  TitleField,
} from "./standard";
import {
  TextWidget,
  SelectWidget,
  CheckboxWidget,
  FileWidget,
  DateWidget,
  TimeWidget,
  DateTimeWidget,
} from "./standard/widgets";

const NoButton = () => null;

export function generateBaseFields(): RegistryFieldsType {
  return {
    AnyOfField,
    ArraySchemaField,
  };
}

export function generateBaseTemplates(): Partial<TemplatesType> {
  return {
    ArrayFieldItemTemplate,
    ArrayFieldTemplate,
    ButtonTemplates: {
      AddButton,
      CopyButton,
      MoveDownButton: NoButton,
      MoveUpButton: NoButton,
      RemoveButton,
      SubmitButton: NoButton,
    },
    DescriptionFieldTemplate: DescriptionField,
    FieldErrorTemplate,
    FieldHelpTemplate,
    FieldTemplate,
    ObjectFieldTemplate,
    OptionalDataControlsTemplate,
    TitleFieldTemplate: TitleField,
    WrapIfAdditionalTemplate,
  };
}

export function generateBaseWidgets(): RegistryWidgetsType {
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

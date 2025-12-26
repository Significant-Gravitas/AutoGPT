import {
  FormContextType,
  RJSFSchema,
  StrictRJSFSchema,
  TemplatesType,
} from "@rjsf/utils";
import { AddButton, CopyButton, RemoveButton } from "../common/buttons";
import { ArrayFieldItemTemplate, ArrayFieldTemplate } from "../array";
import { DescriptionField, FieldErrorTemplate, FieldHelpTemplate, FieldTemplate, TitleField } from "../common/field-templates";
import { ErrorList } from "../common/errors";
import { MultiSchemaFieldTemplate } from "../anyof";
import { ObjectFieldTemplate, OptionalDataControlsTemplate, WrapIfAdditionalTemplate } from "../object";

const NoButton = () => null;

export function generateTemplates<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(): Partial<TemplatesType<T, S, F>> {
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
    // ErrorListTemplate: ErrorList,
    FieldErrorTemplate,
    FieldHelpTemplate,
    FieldTemplate,
    MultiSchemaFieldTemplate,
    ObjectFieldTemplate,
    OptionalDataControlsTemplate,
    TitleFieldTemplate: TitleField,
    WrapIfAdditionalTemplate,
  };
}

export default generateTemplates();

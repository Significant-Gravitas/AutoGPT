import { TemplatesType } from "@rjsf/utils";
import { AddButton, CopyButton, RemoveButton } from "../common/buttons";
import { ArrayFieldItemTemplate, ArrayFieldTemplate } from "../array";
import {
  DescriptionField,
  FieldErrorTemplate,
  FieldHelpTemplate,
  FieldTemplate,
  TitleField,
} from "../common/field-templates";
import {
  ObjectFieldTemplate,
  OptionalDataControlsTemplate,
  WrapIfAdditionalTemplate,
} from "../object";

const NoButton = () => null;

export function generateTemplates(): Partial<TemplatesType> {
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
    ObjectFieldTemplate,
    OptionalDataControlsTemplate,
    TitleFieldTemplate: TitleField,
    WrapIfAdditionalTemplate,
  };
}

export default generateTemplates();

import {
  FormContextType,
  RJSFSchema,
  StrictRJSFSchema,
  TemplatesType,
} from "@rjsf/utils";
import AddButton from "../AddButton";
import ArrayFieldItemTemplate from "../ArrayFieldItemTemplate";
import ArrayFieldTemplate from "../ArrayFieldTemplate";
import DescriptionField from "../DescriptionField";
import ErrorList from "../ErrorList";
import FieldErrorTemplate from "../FieldErrorTemplate";
import FieldHelpTemplate from "../FieldHelpTemplate";
import FieldTemplate from "../FieldTemplate";
import { CopyButton, RemoveButton } from "../IconButton";
import MultiSchemaFieldTemplate from "../MultiSchemaFieldTemplate";
import ObjectFieldTemplate from "../ObjectFieldTemplate";
import OptionalDataControlsTemplate from "../OptionalDataControlsTemplate";
import TitleField from "../TitleField";
import WrapIfAdditionalTemplate from "../WrapIfAdditionalTemplate";

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

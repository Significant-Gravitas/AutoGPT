import ArrayFieldTemplate from "./ArrayFieldTemplate";
import FieldTemplate from "./FieldTemplate";

const NoSubmitButton = () => null;

export const templates = {
  FieldTemplate,
  ButtonTemplates: { SubmitButton: NoSubmitButton },
  ArrayFieldTemplate,
};

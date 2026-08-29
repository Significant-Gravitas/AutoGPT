import { ComponentType } from "react";
import { FormProps, withTheme, ThemeProps } from "@rjsf/core";
import {
  generateBaseFields,
  generateBaseTemplates,
  generateBaseWidgets,
} from "../base/base-registry";
import { generateCustomFields } from "../custom/custom-registry";

export function generateForm(): ComponentType<FormProps> {
  const theme: ThemeProps = {
    templates: generateBaseTemplates(),
    widgets: generateBaseWidgets(),
    fields: {
      ...generateBaseFields(),
      ...generateCustomFields(),
    },
  };

  return withTheme(theme);
}

export default generateForm();

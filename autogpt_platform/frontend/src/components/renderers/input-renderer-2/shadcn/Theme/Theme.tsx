import { ThemeProps } from "@rjsf/core";
import { FormContextType, RJSFSchema, StrictRJSFSchema } from "@rjsf/utils";

import { generateTemplates } from "../Templates";
import { generateWidgets } from "../Widgets";
import { ExtendedFormContextType } from "../types";

export function generateTheme<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = ExtendedFormContextType,
>(): ThemeProps<T, S, F> {
  return {
    templates: generateTemplates<T, S, F>(),
    widgets: generateWidgets<T, S, F>(),
  };
}

export default generateTheme();

import { ComponentType } from "react";

import { FormProps, withTheme } from "@rjsf/core";
import { FormContextType, RJSFSchema, StrictRJSFSchema } from "@rjsf/utils";

import { generateTheme } from "./Theme";
import { ExtendedFormContextType } from "../types";

export function generateForm<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = ExtendedFormContextType,
>(): ComponentType<FormProps<T, S, F>> {
  return withTheme<T, S, F>(generateTheme<T, S, F>());
}

export default generateForm();

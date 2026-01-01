import { ComponentType } from "react";
import { FormProps, withTheme } from "@rjsf/core";
import { generateTheme } from "./Theme";

export function generateForm(): ComponentType<FormProps> {
  return withTheme(generateTheme());
}

export default generateForm();

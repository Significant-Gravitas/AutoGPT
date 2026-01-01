import { ThemeProps } from "@rjsf/core";

import { generateTemplates } from "./templates";
import { generateWidgets } from "./widgets";
import { generateFields } from "./field";

export function generateTheme(): ThemeProps {
  return {
    templates: generateTemplates(),
    widgets: generateWidgets(),
    fields: generateFields(),
  };
}

export default generateTheme();

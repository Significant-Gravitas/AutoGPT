import { withTheme, ThemeProps } from "@rjsf/core";
import {
  generateBaseFields,
  generateBaseTemplates,
  generateBaseWidgets,
} from "../base/base-registry";
import { generateCustomFields } from "../custom/custom-registry";

let theme: ThemeProps | undefined;

function getTheme(): ThemeProps {
  theme ??= {
    templates: generateBaseTemplates(),
    widgets: generateBaseWidgets(),
    fields: {
      ...generateBaseFields(),
      ...generateCustomFields(),
    },
  };
  return theme;
}

// Building the theme reads every field component, so it waits for the form's
// first render: loaded in the middle of an import cycle, this module would run
// before some field exists. withTheme reads the theme only while rendering.
export default withTheme({
  get templates() {
    return getTheme().templates;
  },
  get widgets() {
    return getTheme().widgets;
  },
  get fields() {
    return getTheme().fields;
  },
});

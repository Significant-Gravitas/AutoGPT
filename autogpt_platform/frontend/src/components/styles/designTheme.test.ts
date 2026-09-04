import { Key } from "@/services/storage/local-storage";
import { describe, expect, test } from "vitest";
import {
  DESIGN_THEME_ATTRIBUTE,
  DESIGN_THEME_QUERY_PARAM,
  DESIGN_THEMES,
  designThemeBootScript,
} from "./designTheme";

describe("designThemeBootScript", () => {
  test("stays in sync with the exported constants", () => {
    expect(designThemeBootScript).toContain(`"${Key.DESIGN_THEME}"`);
    expect(designThemeBootScript).toContain(`"${DESIGN_THEME_ATTRIBUTE}"`);
    expect(designThemeBootScript).toContain(
      `get("${DESIGN_THEME_QUERY_PARAM}")`,
    );
    for (const theme of DESIGN_THEMES) {
      expect(designThemeBootScript).toContain(`"${theme}"`);
    }
  });
});

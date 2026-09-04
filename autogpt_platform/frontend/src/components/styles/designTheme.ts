import { Key, storage } from "@/services/storage/local-storage";

export const DESIGN_THEMES = ["default", "linear"] as const;
export type DesignTheme = (typeof DESIGN_THEMES)[number];

export const DESIGN_THEME_ATTRIBUTE = "data-design";
export const DESIGN_THEME_QUERY_PARAM = "design";

function isDesignTheme(value: unknown): value is DesignTheme {
  return DESIGN_THEMES.includes(value as DesignTheme);
}

export function getDesignTheme(): DesignTheme {
  const stored = storage.get(Key.DESIGN_THEME);
  return isDesignTheme(stored) ? stored : "default";
}

export function setDesignTheme(theme: DesignTheme) {
  storage.set(Key.DESIGN_THEME, theme);
  document.documentElement.setAttribute(DESIGN_THEME_ATTRIBUTE, theme);
}

// Runs before hydration so the first paint already has the right tokens.
// `?design=linear` (or `?design=default`) persists the choice; otherwise the
// stored value wins. Written as a static literal (no interpolation) so it is
// never built from runtime values; the key, attribute and theme names must
// match Key.DESIGN_THEME, DESIGN_THEME_ATTRIBUTE and DESIGN_THEMES above.
export const designThemeBootScript = `(function(){try{var k="design-theme";var a="data-design";var ok=["default","linear"];var q=new URLSearchParams(location.search).get("design");if(q&&ok.indexOf(q)>=0){localStorage.setItem(k,q)}var t=localStorage.getItem(k);if(t&&ok.indexOf(t)>=0){document.documentElement.setAttribute(a,t)}}catch(e){}})();`;

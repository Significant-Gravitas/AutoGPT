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
// stored value wins. Kept dependency-free: it is stringified into a <script>.
export const designThemeBootScript = `(function(){try{var k=${JSON.stringify(Key.DESIGN_THEME)};var a=${JSON.stringify(DESIGN_THEME_ATTRIBUTE)};var ok=${JSON.stringify(DESIGN_THEMES)};var q=new URLSearchParams(location.search).get(${JSON.stringify(DESIGN_THEME_QUERY_PARAM)});if(q&&ok.indexOf(q)>=0){localStorage.setItem(k,q)}var t=localStorage.getItem(k);if(t&&ok.indexOf(t)>=0){document.documentElement.setAttribute(a,t)}}catch(e){}})();`;
